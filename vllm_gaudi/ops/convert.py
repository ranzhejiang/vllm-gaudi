import argparse
import json
import os
from glob import glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 240.0  # torch.finfo(torch.float8_e4m3fn).max


def calc_maxabs_scale(xmaxabs, fullscale, backoff=1):
    return xmaxabs / (fullscale * backoff)


def dynamic_quant_per_tensor(data: torch.Tensor, use_unit_quant: bool = False):
    """
    Per-tensor FP8 quantization.
    Returns:
      - fp8_weight: float8_e4m3fn
      - weight_scale: float32 tensor shape=(1,)   <-- KEEP (1,) per your target
    """
    amax = torch.abs(data).max() + 1e-8
    scale = calc_maxabs_scale(amax, FP8_MAX, 1.0)
    if use_unit_quant:
        scale = torch.ones((), device=data.device, dtype=data.dtype)
    scale = scale.to(data.dtype)
    q = torch.clamp(data / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.float().reshape(1)


def copy_other_files(input_path, output_path):
    import shutil

    for file in os.listdir(input_path):
        if file.endswith(".json") or file.endswith(".txt") or file.endswith("jinja"):
            print(f"copying {file} to {output_path}")
            shutil.copyfile(
                os.path.join(input_path, file),
                os.path.join(output_path, file),
            )


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def patch_output_config_quant_scheme(output_path: str):
    """
    Ensure output config has quant_scheme='tensor' and return:
      - num_experts
      - modules_to_not_convert set
    """
    cfg_path = os.path.join(output_path, "config.json")
    cfg = load_json(cfg_path)

    qcfg = cfg.get("quantization_config", {})
    if not qcfg:
        qcfg = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
        }

    qcfg["quant_method"] = qcfg.get("quant_method", "fp8")
    qcfg["fmt"] = qcfg.get("fmt", "e4m3")
    qcfg["quant_scheme"] = "tensor"  # per-tensor

    cfg["quantization_config"] = qcfg
    save_json(cfg_path, cfg)

    num_experts = cfg.get("moe_num_experts", cfg.get("text_config", {}).get("num_experts", 0))
    modules_to_not_convert = set(qcfg.get("modules_to_not_convert", []) or [])
    return num_experts, modules_to_not_convert


def as_fp32_scalar(t: torch.Tensor) -> torch.Tensor:
    """float32 scalar shape=()"""
    return t.to(torch.float32).reshape(())


def is_moe_packed_expert_weight(key: str, tensor: torch.Tensor, num_experts: int) -> bool:
    """
    Leader's heuristic + extra guards:
    - Must be 3D and dim0 == num_experts
    - Must be in ".moe." module
    - Must be one of down/gate/up proj weights
    """
    if not (key.startswith("model.layers.") and ".moe." in key and key.endswith(".weight")):
        return False
    if not (len(tensor.shape) == 3 and tensor.size(0) == num_experts):
        return False
    return any(key.endswith(s) for s in [".down_proj.weight", ".gate_proj.weight", ".up_proj.weight"])


def moe_packed_weight_to_expert_weight_key(packed_weight_key: str, expert_idx: int) -> str:
    """
    Convert:
      model.layers.L.moe.down_proj.weight
    to:
      model.layers.L.moe.experts.<idx>.down_proj.weight
    """
    new_field = f"experts.{expert_idx}"
    return packed_weight_key.replace(".moe.", f".moe.{new_field}.", 1)


def expert_weight_key_to_input_scale_key(expert_weight_key: str) -> str:
    """
    Convert:
      ... .weight
    to:
      ... .input_scale
    """
    assert expert_weight_key.endswith(".weight")
    return expert_weight_key[: -len(".weight")] + ".input_scale"


def convert_files(
    input_path: str,
    output_path: str,
    input_scale_path: str,
    num_experts: int,
    modules_to_not_convert: set[str],
    use_unit_quant: bool,
):
    all_safetensors = glob(f"{input_path}/*.safetensors")
    all_safetensors.sort()
    model_list: dict[str, str] = {}

    with safe_open(input_scale_path, framework="pt", device="cpu") as input_scale:
        # [DEBUG] show a few input_scale keys to confirm naming
        scale_keys = list(input_scale.keys())
        print(f"[DEBUG] input_scale keys total: {len(scale_keys)}")
        for kk in scale_keys[:80]:
            print(f"[DEBUG] input_scale key: {kk}")

        total_quant = 0

        for safetensors_path in all_safetensors:
            print(f"processing {safetensors_path}")
            tensors: dict[str, torch.Tensor] = {}

            with safe_open(safetensors_path, framework="pt", device="cpu") as tensor_file:
                # [DEBUG] show a few weight keys in this shard
                weight_keys = list(tensor_file.keys())
                print(f"[DEBUG] weight shard keys: {len(weight_keys)} total in {os.path.basename(safetensors_path)}")
                for kk in weight_keys[:80]:
                    print(f"[DEBUG] weight key: {kk}")

                for k in tensor_file.keys():
                    tensor = tensor_file.get_tensor(k)

                    # Honor modules_to_not_convert by module name (remove trailing ".weight" if present)
                    k_module = k[:-len(".weight")] if k.endswith(".weight") else k
                    if k_module in modules_to_not_convert:
                        tensors[k] = tensor
                        model_list[k] = os.path.basename(safetensors_path)
                        continue

                    if is_moe_packed_expert_weight(k, tensor, num_experts):
                        # [DEBUG] show shape
                        print(f"[DEBUG] MoE packed weight: {k} shape={tuple(tensor.shape)} dtype={tensor.dtype}")

                        for idx in range(num_experts):
                            expert_tensor = tensor[idx]

                            expert_weight_key = moe_packed_weight_to_expert_weight_key(k, idx)
                            expert_weight_scale_key = expert_weight_key + "_scale"
                            expert_input_scale_key = expert_weight_key_to_input_scale_key(expert_weight_key)

                            if expert_input_scale_key not in input_scale.keys():
                                raise KeyError(
                                    f"Missing input_scale key: {expert_input_scale_key}\n"
                                    f"Derived from packed weight: {k}\n"
                                    f"Expert idx: {idx}"
                                )

                            # Quantize expert weight (per-tensor)
                            w_fp8, w_s = dynamic_quant_per_tensor(expert_tensor, use_unit_quant=use_unit_quant)

                            # Write outputs
                            tensors[expert_weight_key] = w_fp8
                            tensors[expert_weight_scale_key] = w_s  # float32 shape=(1,)

                            # [CHANGED] input_scale -> float32 scalar shape=()
                            tensors[expert_input_scale_key] = as_fp32_scalar(
                                input_scale.get_tensor(expert_input_scale_key).float() * 448.0 / 240.0
                            )

                            shard = os.path.basename(safetensors_path)
                            model_list[expert_weight_key] = shard
                            model_list[expert_weight_scale_key] = shard
                            model_list[expert_input_scale_key] = shard

                            total_quant += 1

                        # IMPORTANT: do NOT keep original packed tensor `k`
                        continue

                    # Otherwise: copy as-is (skip quant)
                    tensors[k] = tensor
                    model_list[k] = os.path.basename(safetensors_path)

            new_tensor_path = safetensors_path.replace(input_path, output_path)
            save_file(tensors, new_tensor_path)
            print(f"saving to {new_tensor_path}")

        print(f"[DEBUG] total quantized expert weights: {total_quant}")

    result = {"weight_map": model_list, "metadata": {}}
    out_json_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Quantize Step-3.5-Flash MoE expert weights to FP8 (per-tensor).")
    parser.add_argument(
        "-i",
        "--input_path",
        default="/mnt/disk3/HF_models/Step-3.5-Flash",
        help="Path to the official model weights.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="/mnt/disk3/HF_models/Step-3.5-Flash-FP8-G2-skip-follow-hy-448",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-s",
        "--input_scale_path",
        default="/mnt/disk6/HF_models/step3p5-input-scale.safetensors",
        help="Path to the input scale safetensors (must match Step-3.5 naming).",
    )
    parser.add_argument(
        "-u",
        "--unit_quant",
        action="store_true",
        help="Enable Unit FP8 Quant (scale=1) for all quantized weights",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_path):
        raise FileNotFoundError(f"input_path not found or not a directory: {args.input_path}")
    if not os.path.isfile(args.input_scale_path):
        raise FileNotFoundError(f"input_scale_path not found: {args.input_scale_path}")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Copy non-weight files (including config.json) first
    copy_other_files(args.input_path, args.output_path)

    # Patch/keep quantization config and load skip list + num_experts
    num_experts, modules_to_not_convert = patch_output_config_quant_scheme(args.output_path)
    print(f"[DEBUG] detected num_experts = {num_experts}")
    print(f"[DEBUG] modules_to_not_convert entries = {len(modules_to_not_convert)}")

    convert_files(
        input_path=args.input_path,
        output_path=args.output_path,
        input_scale_path=args.input_scale_path,
        num_experts=num_experts,
        modules_to_not_convert=modules_to_not_convert,
        use_unit_quant=args.unit_quant,
    )


if __name__ == "__main__":
    main()
