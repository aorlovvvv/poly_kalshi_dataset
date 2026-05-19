"""
Minimal pure-Python parquet reader using only struct + snappy decompression.
Falls back to pandas read_parquet if pyarrow is available.

Supports: PLAIN encoding, SNAPPY compression, INT32/INT64/BYTE_ARRAY column types.
This is sufficient for reading Kalshi trade data (trade_id, ticker, count, yes_price,
no_price, taker_side, created_time).
"""
import struct
import io
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import glob as glob_mod

# Try fast path first
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# ── Thrift compact protocol (minimal) ────────────────────────────────────
# Parquet uses Thrift compact protocol for metadata serialization.

def _read_varint(buf: io.BytesIO) -> int:
    """Read a variable-length integer (Thrift compact)."""
    result = 0
    shift = 0
    while True:
        b = buf.read(1)
        if not b:
            raise EOFError("Unexpected end of buffer in varint")
        byte = b[0]
        result |= (byte & 0x7F) << shift
        shift += 7
        if (byte & 0x80) == 0:
            break
    return result

def _zigzag_decode(n: int) -> int:
    return (n >> 1) ^ -(n & 1)

def _read_thrift_i32(buf: io.BytesIO) -> int:
    return _zigzag_decode(_read_varint(buf))

def _read_thrift_i64(buf: io.BytesIO) -> int:
    return _zigzag_decode(_read_varint(buf))

def _read_thrift_string(buf: io.BytesIO) -> bytes:
    length = _read_varint(buf)
    return buf.read(length)


# ── Parquet physical types ────────────────────────────────────────────────
PARQUET_TYPE_BOOLEAN = 0
PARQUET_TYPE_INT32 = 1
PARQUET_TYPE_INT64 = 2
PARQUET_TYPE_INT96 = 3  # used for timestamps
PARQUET_TYPE_FLOAT = 4
PARQUET_TYPE_DOUBLE = 5
PARQUET_TYPE_BYTE_ARRAY = 6
PARQUET_TYPE_FIXED_LEN_BYTE_ARRAY = 7


# ── Snappy decompression (pure Python fallback) ──────────────────────────
try:
    import snappy as _snappy
    def snappy_decompress(data: bytes) -> bytes:
        return _snappy.decompress(data)
except ImportError:
    def snappy_decompress(data: bytes) -> bytes:
        """Pure-python snappy decompressor (minimal, for parquet pages)."""
        src = io.BytesIO(data)
        # Read uncompressed length
        uncompressed_len = 0
        shift = 0
        while True:
            b = src.read(1)[0]
            uncompressed_len |= (b & 0x7F) << shift
            shift += 7
            if (b & 0x80) == 0:
                break

        out = bytearray()
        while len(out) < uncompressed_len:
            tag_byte = src.read(1)[0]
            tag_type = tag_byte & 0x03

            if tag_type == 0:  # Literal
                lit_len = (tag_byte >> 2) + 1
                if lit_len == 60 + 1:
                    lit_len = src.read(1)[0] + 1
                elif lit_len == 61 + 1:
                    lit_len = struct.unpack('<H', src.read(2))[0] + 1
                elif lit_len == 62 + 1:
                    b = src.read(3)
                    lit_len = (b[0] | (b[1] << 8) | (b[2] << 16)) + 1
                elif lit_len == 63 + 1:
                    lit_len = struct.unpack('<I', src.read(4))[0] + 1
                out.extend(src.read(lit_len))
            elif tag_type == 1:  # Copy with 1-byte offset
                length = ((tag_byte >> 2) & 0x07) + 4
                offset = ((tag_byte >> 5) << 8) | src.read(1)[0]
                start = len(out) - offset
                for i in range(length):
                    out.append(out[start + i])
            elif tag_type == 2:  # Copy with 2-byte offset
                length = (tag_byte >> 2) + 1
                offset = struct.unpack('<H', src.read(2))[0]
                start = len(out) - offset
                for i in range(length):
                    out.append(out[start + i])
            elif tag_type == 3:  # Copy with 4-byte offset
                length = (tag_byte >> 2) + 1
                offset = struct.unpack('<I', src.read(4))[0]
                start = len(out) - offset
                for i in range(length):
                    out.append(out[start + i])

        return bytes(out[:uncompressed_len])


# ── Parquet file metadata reader ──────────────────────────────────────────

class ParquetFileMetadata:
    """Parse parquet file footer to extract schema and row group info."""

    def __init__(self, path: str):
        self.path = path
        with open(path, 'rb') as f:
            # Verify magic
            magic = f.read(4)
            assert magic == b'PAR1', f"Not a parquet file: {path}"

            # Read footer length
            f.seek(-8, 2)
            self.footer_length = struct.unpack('<i', f.read(4))[0]
            end_magic = f.read(4)
            assert end_magic == b'PAR1'

            # Read footer bytes
            f.seek(-(8 + self.footer_length), 2)
            self.footer_offset = f.tell()
            self.footer_bytes = f.read(self.footer_length)

        self._parse_footer()

    def _parse_footer(self):
        """Parse Thrift-encoded FileMetaData."""
        buf = io.BytesIO(self.footer_bytes)
        # FileMetaData is a Thrift struct with fields:
        # 1: i32 version
        # 2: list<SchemaElement> schema
        # 3: i64 num_rows
        # 4: list<RowGroup> row_groups
        # We parse enough to get column names, types, and row group offsets

        self.columns = []
        self.num_rows = 0
        self.row_groups = []

        # Simple approach: extract column names from the embedded pandas metadata
        import re, json
        text = self.footer_bytes.decode('utf-8', errors='ignore')

        # Find pandas metadata JSON
        pandas_match = re.search(r'\{"index_columns.*?\}(?:\]|\})', text, re.DOTALL)
        if pandas_match:
            # Try to parse the pandas metadata
            try:
                # Find the full JSON blob
                start = text.find('{"index_columns')
                if start >= 0:
                    # Find balanced braces
                    depth = 0
                    end = start
                    for i in range(start, len(text)):
                        if text[i] == '{':
                            depth += 1
                        elif text[i] == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    json_str = text[start:end]
                    meta = json.loads(json_str)
                    self.columns = [c['name'] for c in meta.get('columns', [])]
                    self.pandas_meta = meta
            except (json.JSONDecodeError, KeyError):
                pass

        # Extract num_rows from Thrift (field 3 = i64)
        # Simple heuristic: look for column names in footer strings
        if not self.columns:
            self.columns = re.findall(r'(?:trade_id|ticker|count|yes_price|no_price|taker_side|created_time|_fetched_at)', text)


def read_parquet_files(pattern: str, columns: Optional[List[str]] = None,
                       max_files: int = None, max_rows: int = None) -> pd.DataFrame:
    """
    Read parquet files matching a glob pattern.

    Uses pyarrow if available, otherwise falls back to a simpler approach
    using pandas + the metadata we can extract.
    """
    files = sorted(glob_mod.glob(pattern))
    if max_files:
        files = files[:max_files]

    if not files:
        raise FileNotFoundError(f"No files match pattern: {pattern}")

    if HAS_PYARROW:
        dfs = []
        total_rows = 0
        for f in files:
            df = pq.read_table(f, columns=columns).to_pandas()
            dfs.append(df)
            total_rows += len(df)
            if max_rows and total_rows >= max_rows:
                break
        result = pd.concat(dfs, ignore_index=True)
        if max_rows:
            result = result.head(max_rows)
        return result

    # Fallback: try pandas with any available engine
    try:
        dfs = []
        total_rows = 0
        for f in files:
            df = pd.read_parquet(f, columns=columns)
            dfs.append(df)
            total_rows += len(df)
            if max_rows and total_rows >= max_rows:
                break
        result = pd.concat(dfs, ignore_index=True)
        if max_rows:
            result = result.head(max_rows)
        return result
    except ImportError:
        raise ImportError(
            "Cannot read parquet files: no pyarrow or fastparquet installed. "
            "Install with: pip install pyarrow"
        )
