import json
from pathlib import Path
from typing import Callable

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name.lower() not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name.lower()]


def _is_client_key(key: str, split_layer: int) -> bool:
    if key.startswith("model.embed_tokens."):
        return True
    if key.startswith("model.rotary_emb."):
        return True
    for idx in range(split_layer):
        if key.startswith(f"model.layers.{idx}."):
            return True
    return False


def _is_server_key(key: str, split_layer: int, num_layers: int) -> bool:
    if key.startswith("model.embed_tokens."):
        return True
    if key.startswith("model.norm.") or key.startswith("lm_head."):
        return True
    if key.startswith("model.rotary_emb."):
        return True
    for idx in range(split_layer, num_layers):
        if key.startswith(f"model.layers.{idx}."):
            return True
    return False


def _load_from_single_safetensor(
    model,
    safetensor_file: Path,
    should_load: Callable[[str], bool],
    dtype: torch.dtype,
):
    with safe_open(str(safetensor_file), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not should_load(key):
                continue
            tensor = handle.get_tensor(key).to(dtype=dtype)
            set_module_tensor_to_device(model, key, "cpu", value=tensor)


def _load_from_indexed_safetensors(
    model,
    index_file: Path,
    should_load: Callable[[str], bool],
    dtype: torch.dtype,
):
    with index_file.open("r", encoding="utf-8") as handle:
        index_data = json.load(handle)

    weight_map = index_data["weight_map"]
    by_file = {}
    for key, filename in weight_map.items():
        if should_load(key):
            by_file.setdefault(filename, []).append(key)

    for filename, keys in by_file.items():
        shard_path = index_file.parent / filename
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in keys:
                tensor = handle.get_tensor(key).to(dtype=dtype)
                set_module_tensor_to_device(model, key, "cpu", value=tensor)


def load_partial_model(model_name: str, split_layer: int, role: str, dtype_name: str):
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    dtype = _dtype_from_name(dtype_name)
    config.torch_dtype = dtype

    if role not in {"client", "server"}:
        raise ValueError(f"Unknown role: {role}")
    if not (0 < split_layer < num_layers):
        raise ValueError(
            f"split_layer must be in [1, {num_layers - 1}], got {split_layer}"
        )

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    if role == "client":

        def should_load(key: str) -> bool:
            return _is_client_key(key, split_layer)

    else:

        def should_load(key: str) -> bool:
            return _is_server_key(key, split_layer, num_layers)

    snapshot_dir = Path(
        snapshot_download(
            repo_id=model_name,
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors"],
        )
    )

    index_file = snapshot_dir / "model.safetensors.index.json"
    single_file = snapshot_dir / "model.safetensors"

    if index_file.exists():
        _load_from_indexed_safetensors(model, index_file, should_load, dtype)
    elif single_file.exists():
        _load_from_single_safetensor(model, single_file, should_load, dtype)
    else:
        raise FileNotFoundError("No safetensors checkpoint found in local snapshot.")

    if role == "server" and hasattr(model, "lm_head") and model.lm_head.weight.is_meta:
        model.tie_weights()

    model.eval()
    return model
