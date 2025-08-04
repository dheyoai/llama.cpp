import numpy as np
import struct
def float_to_e5m2funz_byte(f: float) -> int:
    """A standard-compliant float32 to E5M2FUNZ byte conversion."""
    u32 = struct.unpack('!I', struct.pack('!f', f))[0]
    sign = (u32 >> 31)
    exp32 = (u32 >> 23) & 0xFF
    mant32 = u32 & 0x7FFFFF
    if exp32 == 0xFF:
        return (sign << 7) | 0b0_11111_00 if mant32 == 0 else 0b0_11111_01
    exp5 = exp32 - 112
    if exp32 == 0 or exp5 <= 0: return (sign << 7) | 0
    if exp5 >= 31: return (sign << 7) | 0b0_11111_00
    rounding_bit_pos = 23 - 2 - 1
    mant32 += (1 << rounding_bit_pos)
    if mant32 & (1 << 24):
        exp5 += 1
        mant32 = 0
        if exp5 >= 31: return (sign << 7) | 0b0_11111_00
    mant2 = (mant32 >> (23 - 2)) & 0b11
    return (sign << 7) | (exp5 << 2) | mant2
def quantize_mx_interleaved_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    max_abs_vals = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.divide(max_abs_vals, 1.75, out=np.zeros_like(max_abs_vals), where=max_abs_vals!=0).astype(np.float16)
    normalized_blocks = np.divide(blocks, scales, out=np.zeros_like(blocks), where=scales!=0)
    u32 = normalized_blocks.view(np.uint32)
    sign = (u32 >> 31).astype(np.uint8)
    exp32 = ((u32 >> 23) & 0xFF).astype(np.int16)
    mant32 = (u32 & 0x7FFFFF)
    is_special = (exp32 == 0xFF)
    special_vals = np.where(mant32 == 0, (sign << 7) | 0x7C, 0x7D)
    exp5 = exp32 - 112
    is_ftz_or_overflow = (exp32 == 0) | (exp5 <= 0) | (exp5 >= 31)
    rounding_bit_pos = 23 - 2 - 1
    mant32 += (1 << rounding_bit_pos)
    overflowed_mant = (mant32 & (1 << 24)) != 0
    exp5[overflowed_mant] += 1
    mant32[overflowed_mant] = 0
    is_ftz_or_overflow |= (exp5 >= 31)

    mant2 = ((mant32 >> (23 - 2)) & 0b11).astype(np.uint8)
    quantized_qs = (sign << 7) | (exp5.astype(np.uint8) << 2) | mant2
    quantized_qs[is_special] = special_vals[is_special]
    quantized_qs[is_ftz_or_overflow] = (sign << 7)[is_ftz_or_overflow]
    quantized_qs = quantized_qs.reshape(n_blocks, 32)
    scales_bytes = scales.view(np.uint8).reshape(n_blocks, 2)
    packed_array = np.concatenate([scales_bytes, quantized_qs], axis=1)
    return packed_array

