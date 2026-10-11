"""
图片感知哈希：用于缓存键生成。
采用 pHash（感知哈希），对 JPEG 压缩、轻微缩放有一定鲁棒性，
同时保证不同内容图片哈希不同。
"""
from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np
from PIL import Image


def _dct_2d(arr: np.ndarray) -> np.ndarray:
    """简易 2D DCT（基于 FFT 实现，避免引入 scipy 依赖）。"""
    M, N = arr.shape
    # 对行做 DCT
    arr_ext = np.hstack([arr, np.fliplr(arr[:, 1:-1])])
    freq_row = np.fft.fft(arr_ext, axis=1)[:, :N]
    # 对列做 DCT
    freq_ext = np.vstack([freq_row, np.flipud(freq_row[1:-1, :])])
    freq = np.fft.fft(freq_ext, axis=0)[:M, :]
    return np.real(freq)


def phash(img: Image.Image, hash_size: int = 16, highfreq_factor: int = 4) -> str:
    """
    感知哈希：返回 16 进制字符串。
    算法：缩放 -> 灰度 -> DCT -> 取左上角低频 -> 与均值比较生成二值串。
    """
    img_size = hash_size * highfreq_factor
    gray = img.convert("L").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    dct = _dct_2d(arr)
    low_freq = dct[:hash_size, :hash_size]
    # 排除直流分量（0,0），取其余均值
    median = np.median(low_freq[1:])
    bits = (low_freq > median).flatten()
    # 转为 16 进制
    bit_string = "".join("1" if b else "0" for b in bits)
    hex_str = f"{int(bit_string, 2):0{hash_size * hash_size // 4}x}"
    return hex_str


def md5_hash(data: bytes) -> str:
    """精确哈希：用于完全相同的重复请求。"""
    return hashlib.md5(data).hexdigest()


def combined_cache_key(img: Image.Image, raw_bytes: bytes) -> str:
    """
    组合缓存键：md5(精确匹配) + phash(感知匹配)。
    实际缓存以 md5 为主键，phash 作为辅助索引用于近似命中（预留）。
    """
    return f"img:{md5_hash(raw_bytes)}"


def hamming_distance(hash1: str, hash2: str) -> int:
    """两个 16 进制哈希串的汉明距离。"""
    int1 = int(hash1, 16)
    int2 = int(hash2, 16)
    return bin(int1 ^ int2).count("1")
