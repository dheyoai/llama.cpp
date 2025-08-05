#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import struct

# Path adjustment for running as a module
if __package__ is None and (Path(__file__).parent.parent.parent / 'gguf').is_dir():
    sys.path.append(str(Path(__file__).parent.parent.resolve()))

try:
    # We only need the constants to map the enum values back to names
    from gguf.constants import GGMLQuantizationType
except ImportError as e:
    print(f"Error: {e}. The script could not find the gguf library.", file=sys.stderr)
    sys.exit(1)

# --- GGUF Parsing Helpers ---

def read_string(f):
    """Reads a GGUF-encoded string from a file object."""
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors='ignore')

def get_ggml_type_name(dtype_enum: int) -> str:
    """Converts a GGML quantization enum value to its string name."""
    try:
        return GGMLQuantizationType(dtype_enum).name
    except ValueError:
        return f"UNKNOWN({dtype_enum})"

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m gguf.scripts.gguf_inspect <path_to_gguf_file>")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Error: File not found at {filepath}")
        sys.exit(1)

    print(f"--- Inspecting GGUF File: {filepath.name} ---")

    with open(filepath, "rb") as f:
        # 1. Read Header
        magic, version, tensor_count, kv_count = struct.unpack("<IIQQ", f.read(24))
        
        if magic != 0x46554747: # "GGUF"
            print("Error: This is not a valid GGUF file (magic number mismatch).")
            return
            
        print(f"GGUF Version: {version}, Tensor Count: {tensor_count}, Metadata KV Count: {kv_count}\n")

        # 2. Skip the detailed metadata section for this simple tool
        for _ in range(kv_count):
            read_string(f) # key
            (value_type_enum,) = struct.unpack("<I", f.read(4))
            # This is a simplified skipping logic that may not cover all complex types
            if value_type_enum == 8: # String
                val_len = struct.unpack("<Q", f.read(8))[0]
                f.seek(val_len, 1)
            elif value_type_enum == 9: # Array
                f.seek(4, 1) # array type
                arr_len = struct.unpack("<Q", f.read(8))[0]
                for _ in range(arr_len):
                    str_len = struct.unpack("<Q", f.read(8))[0]
                    f.seek(str_len, 1)
            elif value_type_enum in [0, 1, 7]: f.seek(1, 1)
            elif value_type_enum in [2, 3]: f.seek(2, 1)
            elif value_type_enum in [4, 5, 6]: f.seek(4, 1)
            elif value_type_enum in [10, 11, 12]: f.seek(8, 1)

        # 3. Read and Print Tensor Info
        print(f'--- Tensors ({tensor_count} total) ---')
        print(f'  {"#":>3} | {"Name":<60} | {"Shape":<20} | {"Data Format (dtype)"}')
        print(f'  {"-"*3} | {"-"*60} | {"-"*20} | {"-"*20}')

        for i in range(tensor_count):
            name = read_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            # Shape is stored in reverse order in GGUF
            shape = list(reversed(struct.unpack(f"<{n_dims}Q", f.read(n_dims * 8))))
            (dtype_enum,) = struct.unpack("<I", f.read(4))
            f.seek(8, 1) # Skip the offset, we don't need it for this tool
            
            shape_str = ", ".join(map(str, shape))
            dtype_name = get_ggml_type_name(dtype_enum)
            
            # Print the formatted row
            print(f'  {i+1:>3} | {name:<60} | {shape_str:<20} | {dtype_name}')

if __name__ == "__main__":
    main()