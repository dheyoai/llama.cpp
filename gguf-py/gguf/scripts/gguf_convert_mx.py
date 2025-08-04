#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import struct
import json

if __package__ is None and (Path(__file__).parent.parent.parent / 'gguf').is_dir():
    sys.path.append(str(Path(__file__).parent.parent.resolve()))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import numpy as np
    from gguf.constants import GGUF_MAGIC, GGUF_VERSION, GGUFValueType, GGMLQuantizationType, Keys
    from gguf.mx_packer import quantize_mx_interleaved_blocks
except ImportError as e:
    print(f"Error: {e}. Please run 'pip install transformers torch sentencepiece'", file=sys.stderr)
    sys.exit(1)

def write_string(f, s: str):
    encoded = s.encode('utf8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)

def write_kv_item(f, key: str, value, value_type: GGUFValueType):
    write_string(f, key)
    f.write(struct.pack('<I', value_type.value))
    if value_type == GGUFValueType.UINT32:
        f.write(struct.pack('<I', value))
    elif value_type == GGUFValueType.FLOAT32:
        f.write(struct.pack('<f', value))
    elif value_type == GGUFValueType.BOOL:
        f.write(struct.pack('<?', value))
    elif value_type == GGUFValueType.STRING:
        write_string(f, value)
    elif value_type == GGUFValueType.ARRAY:
        f.write(struct.pack('<I', GGUFValueType.STRING.value))
        f.write(struct.pack('<Q', len(value)))
        for item in value:
            write_string(f, item)
    else:
        raise NotImplementedError(f"Type {value_type} not implemented in mini writer")

# --- NEW Tensor Name Mapping for Qwen/DeepSeek Architecture ---
def map_qwen_tensor_name(hf_name: str, block_count: int) -> str:
    if "model.embed_tokens.weight" in hf_name: return "token_embd.weight"
    if "lm_head.weight" in hf_name: return "output.weight"
    if "model.norm.weight" in hf_name: return "output_norm.weight"

    if "model.layers" in hf_name:
        try:
            block_id = int(hf_name.split('.')[2])
        except (ValueError, IndexError):
            return "" # Skip
        
        # Attention blocks
        if f"layers.{block_id}.self_attn.q_proj.weight" in hf_name: return f"blk.{block_id}.attn_q.weight"
        if f"layers.{block_id}.self_attn.k_proj.weight" in hf_name: return f"blk.{block_id}.attn_k.weight"
        if f"layers.{block_id}.self_attn.v_proj.weight" in hf_name: return f"blk.{block_id}.attn_v.weight"
        if f"layers.{block_id}.self_attn.o_proj.weight" in hf_name: return f"blk.{block_id}.attn_output.weight"
        if f"layers.{block_id}.input_layernorm.weight" in hf_name: return f"blk.{block_id}.attn_norm.weight"
        
        # MLP blocks
        if f"layers.{block_id}.mlp.gate_proj.weight" in hf_name: return f"blk.{block_id}.ffn_gate.weight"
        if f"layers.{block_id}.mlp.up_proj.weight" in hf_name: return f"blk.{block_id}.ffn_up.weight"
        if f"layers.{block_id}.mlp.down_proj.weight" in hf_name: return f"blk.{block_id}.ffn_down.weight"
        if f"layers.{block_id}.post_attention_layernorm.weight" in hf_name: return f"blk.{block_id}.ffn_norm.weight"

    return "" # Skip biases and other tensors for this example

def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    output_path = Path(f"./{model_name.split('/')[-1]}-MX_INTERLEAVED.gguf")
    qtype = GGMLQuantizationType.MX_E5M2_B32_INTERLEAVED

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = model.config

    # --- Prepare Metadata (adapted for Qwen/DeepSeek) ---
    metadata = {
        Keys.General.ARCHITECTURE: "qwen2",
        Keys.General.NAME: model_name,
        Keys.LLM.CONTEXT_LENGTH.format(arch="qwen2"): config.max_position_embeddings,
        Keys.LLM.EMBEDDING_LENGTH.format(arch="qwen2"): config.hidden_size,
        Keys.LLM.BLOCK_COUNT.format(arch="qwen2"): config.num_hidden_layers,
        Keys.LLM.FEED_FORWARD_LENGTH.format(arch="qwen2"): config.intermediate_size,
        Keys.Attention.HEAD_COUNT.format(arch="qwen2"): config.num_attention_heads,
        Keys.Attention.HEAD_COUNT_KV.format(arch="qwen2"): config.num_key_value_heads,
        Keys.Attention.LAYERNORM_RMS_EPS.format(arch="qwen2"): config.rms_norm_eps,
        # --- THIS IS THE CORRECTED LINE ---
        Keys.Rope.DIMENSION_COUNT.format(arch="qwen2"): int(config.hidden_size / config.num_attention_heads),
        Keys.Rope.FREQ_BASE.format(arch="qwen2"): config.rope_theta,
        Keys.Tokenizer.MODEL: "qwen",
        Keys.Tokenizer.BOS_ID: tokenizer.bos_token_id,
        Keys.Tokenizer.EOS_ID: tokenizer.eos_token_id,
    }
    
    # --- The rest of the function is unchanged ---
    
    vocab = [tokenizer.decode([i]) for i in range(len(tokenizer.get_vocab()))]
    
    tensors_to_write = []
    state_dict = model.state_dict()
    for hf_name, tensor in state_dict.items():
        gguf_name = map_qwen_tensor_name(hf_name, config.num_hidden_layers)
        if not gguf_name:
            print(f"Skipping tensor {hf_name}")
            continue
        
        if tensor.ndim == 2 and "weight" in gguf_name:
            if tensor.shape[0] % 32 != 0:
                pad_width = 32 - (tensor.shape[0] % 32)
                tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_width))
                print(f"Padding tensor {hf_name} from {tensor.shape[0]-pad_width} to {tensor.shape[0]}")

            print(f"Quantizing tensor {hf_name} -> {gguf_name}")
            flat_tensor = tensor.to(torch.float32).numpy().flatten()
            blocks = flat_tensor.reshape(-1, 32)
            quantized_data = quantize_mx_interleaved_blocks(blocks).tobytes()
            tensors_to_write.append({
                "name": gguf_name, "data": quantized_data,
                "shape": tensor.shape, "dtype": qtype,
            })
        else:
            print(f"Storing tensor {hf_name} -> {gguf_name} as F32")
            tensors_to_write.append({
                "name": gguf_name, "data": tensor.to(torch.float32).numpy().tobytes(),
                "shape": tensor.shape, "dtype": GGMLQuantizationType.F32,
            })

    print(f"\nWriting to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(tensors_to_write)))
        f.write(struct.pack('<Q', len(metadata) + 1))

        for key, value in metadata.items():
            if isinstance(value, str): write_kv_item(f, key, value, GGUFValueType.STRING)
            elif isinstance(value, int): write_kv_item(f, key, value, GGUFValueType.UINT32)
            elif isinstance(value, float): write_kv_item(f, key, value, GGUFValueType.FLOAT32)
            elif isinstance(value, bool): write_kv_item(f, key, value, GGUFValueType.BOOL)
        
        write_kv_item(f, Keys.Tokenizer.LIST, vocab, GGUFValueType.ARRAY)

        tensor_data_offset = 0
        for tensor in tensors_to_write:
            write_string(f, tensor["name"])
            shape = tensor["shape"]
            f.write(struct.pack('<I', len(shape)))
            for dim in reversed(shape): f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', tensor["dtype"].value))
            f.write(struct.pack('<Q', tensor_data_offset))
            tensor_data_offset += len(tensor["data"])
        
        offset = f.tell()
        align = 32
        padding = (align - (offset % align)) % align
        f.write(b'\x00' * padding)
        
        for tensor in tensors_to_write:
            f.write(tensor["data"])

    print(f"Successfully created GGUF file: {output_path}")

if __name__ == '__main__':
    main()