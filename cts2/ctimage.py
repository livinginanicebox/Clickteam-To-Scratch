"""Decode Clickteam Fusion image bitmaps (uncompressed and RLE/RLEW/RLET modes).

Every function here works on *whole rows* (or whole images) instead of
walking pixels: a game bank holds thousands of bitmaps, and a per-pixel
Python loop over a 320x240 image already costs tens of milliseconds — enough
to make a conversion look frozen on "decoding images".  Extended-slice
assignment, ``bytes.translate`` and big-integer ``and``/``or`` do the same
work at C speed.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .png import encode_png


def _align(value: int, align: int) -> int:
    rem = value % align
    return value if rem == 0 else value + (align - rem)


# --------------------------------------------------------------------------
# RLE / RLEW / RLET decompression
# --------------------------------------------------------------------------

def decompress_rle(data: bytes, unit_size: int,
                   max_bytes: int) -> Tuple[bytes, int]:
    """Decompress Clickteam RLE (unit=1), RLEW (unit=2), or RLET (unit=3).

    Returns ``(decompressed_bytes, bytes_consumed_from_data)``.
    Stream format:
      - ``cmd == 0``: end of stream
      - ``cmd > 128``: literal run of ``cmd - 128`` units
      - ``1 <= cmd <= 128``: repeat run of ``cmd`` copies of the next unit
    """
    out = bytearray()
    pos = 0
    n = len(data)
    while pos < n:
        cmd = data[pos]
        pos += 1
        if cmd == 0:
            break
        if cmd > 128:
            count = cmd - 128
            need = count * unit_size
            if pos + need > n:
                avail = (n - pos) // unit_size
                if avail > 0:
                    out.extend(data[pos:pos + avail * unit_size])
                pos = n
                break
            out.extend(data[pos:pos + need])
            pos += need
        else:
            if pos + unit_size > n:
                pos = n
                break
            unit = data[pos:pos + unit_size]
            pos += unit_size
            out.extend(unit * cmd)
        if max_bytes > 0 and len(out) >= max_bytes:
            if pos < n and data[pos] == 0:
                pos += 1
            break
    return bytes(out), pos


# --------------------------------------------------------------------------
# buffer helpers (whole-image, no per-pixel loop)
# --------------------------------------------------------------------------

def _gather_rows(data: bytes, height: int, row_bytes: int,
                 stride: int) -> bytes:
    """Concatenate ``height`` scanlines of ``data``, dropping row padding."""
    needed = (height - 1) * stride + row_bytes
    if needed > len(data):
        raise IndexError(
            f"bitmap truncated: need {needed} bytes, have {len(data)}")
    if stride == row_bytes:
        return data[: height * row_bytes]        # no padding: one slice
    return b"".join(data[y * stride: y * stride + row_bytes]
                    for y in range(height))


def _pick_stride(data_len: int, height: int, row_bytes: int,
                 preferred_align: int = 2) -> int:
    """Choose the scanline stride that fits ``data_len``."""
    candidates = list(dict.fromkeys([
        _align(row_bytes, preferred_align),
        _align(row_bytes, 4),
        row_bytes,
    ]))
    for stride in candidates:
        if stride * height == data_len:
            return stride
    for stride in candidates:
        if (height - 1) * stride + row_bytes <= data_len:
            return stride
    return candidates[0]


_SLICE = 1 << 22


def _combine_bytes(a: bytes, b: bytes, op) -> bytes:
    """``op`` (``or`` / ``and``) applied per byte, in C-speed chunks."""
    n = len(a)
    if n != len(b):
        raise ValueError("mismatched buffer sizes")
    if n == 0:
        return b""
    if n <= _SLICE:
        return op(int.from_bytes(a, "little"),
                  int.from_bytes(b, "little")).to_bytes(n, "little")
    pieces = []
    for pos in range(0, n, _SLICE):
        size = min(_SLICE, n - pos)
        pieces.append(op(int.from_bytes(a[pos:pos + size], "little"),
                         int.from_bytes(b[pos:pos + size], "little")
                         ).to_bytes(size, "little"))
    return b"".join(pieces)


def _or_bytes(a: bytes, b: bytes) -> bytes:
    return _combine_bytes(a, b, lambda x, y: x | y)


def _and_bytes(a: bytes, b: bytes) -> bytes:
    return _combine_bytes(a, b, lambda x, y: x & y)


def _table(fn) -> bytes:
    """256-entry ``bytes.translate`` table."""
    return bytes(fn(v) & 0xFF for v in range(256))


def _eq_table(value: int) -> bytes:
    """Translate table: 0xFF where the byte equals ``value``, else 0x00."""
    return bytes(0xFF if v == value else 0x00 for v in range(256))


_INVERT = _table(lambda v: (~v) & 0xFF)

# 5-5-5 (15 bpp) and 5-6-5 (16 bpp) channel expansions, as translate tables
# over the low/high byte of every pixel.
_T_R15 = _table(lambda v: (v << 1) & 0xF8)
_T_G15_LO = _table(lambda v: (v & 0xE0) >> 2)
_T_G15_HI = _table(lambda v: (v & 0x03) << 6)
_T_R16 = _table(lambda v: v & 0xF8)
_T_G16_LO = _table(lambda v: (v & 0xE0) >> 3)
_T_G16_HI = _table(lambda v: (v & 0x07) << 5)
_T_BLUE5 = _table(lambda v: (v & 0x1F) << 3)


# --------------------------------------------------------------------------
# pixel formats
# --------------------------------------------------------------------------

def read_24bpp(data: bytes, width: int, height: int) -> tuple:
    """BGR triplets, padded to 2 bytes (or unpadded for RLET)."""
    n = width * height
    row_bytes = width * 3
    stride = _pick_stride(len(data), height, row_bytes, preferred_align=2)
    body = _gather_rows(data, height, row_bytes, stride)
    out = bytearray(n * 4)
    out[0::4] = body[2::3]          # R
    out[1::4] = body[1::3]          # G
    out[2::4] = body[0::3]          # B
    out[3::4] = b"\xff" * n         # A
    return bytes(out), stride * height


def read_15bpp(data: bytes, width: int, height: int) -> tuple:
    n = width * height
    row_bytes = width * 2
    stride = _pick_stride(len(data), height, row_bytes, preferred_align=2)
    body = _gather_rows(data, height, row_bytes, stride)
    lo = body[0::2]
    hi = body[1::2]
    out = bytearray(n * 4)
    out[0::4] = hi.translate(_T_R15)
    out[1::4] = _or_bytes(lo.translate(_T_G15_LO), hi.translate(_T_G15_HI))
    out[2::4] = lo.translate(_T_BLUE5)
    out[3::4] = b"\xff" * n
    return bytes(out), stride * height


def read_16bpp(data: bytes, width: int, height: int) -> tuple:
    n = width * height
    row_bytes = width * 2
    stride = _pick_stride(len(data), height, row_bytes, preferred_align=2)
    body = _gather_rows(data, height, row_bytes, stride)
    lo = body[0::2]
    hi = body[1::2]
    out = bytearray(n * 4)
    out[0::4] = hi.translate(_T_R16)
    out[1::4] = _or_bytes(lo.translate(_T_G16_LO), hi.translate(_T_G16_HI))
    out[2::4] = lo.translate(_T_BLUE5)
    out[3::4] = b"\xff" * n
    return bytes(out), stride * height


def read_32bpp(data: bytes, width: int, height: int) -> tuple:
    row_bytes = width * 4
    stride = _pick_stride(len(data), height, row_bytes, preferred_align=4)
    body = _gather_rows(data, height, row_bytes, stride)
    out = bytearray(body)
    out[0::4] = body[2::4]          # R <- third stored byte
    out[2::4] = body[0::4]          # B <- first stored byte
    # G and the stored alpha byte keep their positions.
    return bytes(out), stride * height


def read_8bpp(data: bytes, width: int, height: int,
              palette=None) -> tuple:
    """8-bit indexed (mode 3) bitmap, using palette or grayscale fallback."""
    n = width * height
    stride = _pick_stride(len(data), height, width, preferred_align=2)
    body = _gather_rows(data, height, width, stride)
    if palette and len(palette) >= 256:
        rt = bytes(palette[i][0] & 0xFF for i in range(256))
        gt = bytes(palette[i][1] & 0xFF for i in range(256))
        bt = bytes(palette[i][2] & 0xFF for i in range(256))
        out = bytearray(n * 4)
        out[0::4] = body.translate(rt)
        out[1::4] = body.translate(gt)
        out[2::4] = body.translate(bt)
        out[3::4] = b"\xff" * n
        return bytes(out), stride * height
    out = bytearray(n * 4)
    out[0::4] = body
    out[1::4] = body
    out[2::4] = body
    out[3::4] = b"\xff" * n
    return bytes(out), stride * height


def read_alpha(data: bytes, width: int, height: int,
               pos: int) -> Optional[bytes]:
    """The separate 8-bit alpha plane that follows the pixel data."""
    buf = data[pos:] if pos else data
    if not buf:
        return None
    for align in (4, 2, 1):
        stride = _align(width, align)
        if stride * height == len(buf):
            return _gather_rows(buf, height, width, stride)
    for align in (4, 2, 1):
        stride = _align(width, align)
        if (height - 1) * stride + width <= len(buf):
            return _gather_rows(buf, height, width, stride)
    return None


def _color_key_match(rgba: bytes, width: int, height: int,
                     transparent: tuple) -> bytes:
    """One byte per pixel: 0xFF where the RGB equals the transparent colour."""
    r = rgba[0::4]
    g = rgba[1::4]
    b = rgba[2::4]
    tr, tg, tb = transparent[0], transparent[1], transparent[2]
    return _and_bytes(
        _and_bytes(r.translate(_eq_table(tr)), g.translate(_eq_table(tg))),
        b.translate(_eq_table(tb)))


def decode_bmp(width: int, height: int, mode: int, flags: int, body: bytes,
               transparent=None, palette=None) -> Optional[bytes]:
    """Return PNG bytes for a Clickteam bitmap.

    flags bits: bit0 RLE, bit1 RLEW, bit2 RLET, bit3 LZX, bit4 Alpha,
    bit7 RGBA.
    """
    if width <= 0 or height <= 0 or not body:
        return None
    if flags & 0x08:
        return None

    mode = mode & 0xFF
    bpp_map = {0: 4, 3: 1, 4: 3, 6: 2, 7: 2, 8: 4, 16: 4}
    bpp = bpp_map.get(mode)
    if bpp is None:
        return None

    alpha_buf: Optional[bytes] = None
    color_data = body
    if flags & 0x07:
        if flags & 0x04:
            unit_size = 3
        elif flags & 0x02:
            unit_size = 2
        else:
            unit_size = 1
        max_color = _align(width * bpp, 4) * height
        color_data, pos = decompress_rle(body, unit_size, max_color)
        if len(color_data) < width * height * bpp:
            return None
        if flags & 0x10:
            rest = body[pos:]
            max_alpha = _align(width, 4) * height
            alpha_rle, _ = decompress_rle(rest, 1, max_alpha)
            if len(alpha_rle) >= width * height:
                alpha_buf = alpha_rle
            elif len(rest) >= width * height:
                alpha_buf = rest
            else:
                alpha_buf = b""

    if mode == 3:
        rgba, used = read_8bpp(color_data, width, height, palette=palette)
    elif mode == 4:
        rgba, used = read_24bpp(color_data, width, height)
    elif mode == 6:
        rgba, used = read_15bpp(color_data, width, height)
    elif mode == 7:
        rgba, used = read_16bpp(color_data, width, height)
    elif mode == 8:
        # Fusion 2.5+ EXEs: 32 bits per pixel, 8 bits per channel (BGRA).
        rgba, used = read_32bpp(color_data, width, height)
        if flags & 0x80:  # RGBA flag: the 4th byte is the alpha channel
            if (flags & 0x10) and rgba[3::4].count(0) == width * height:
                a_src = alpha_buf if alpha_buf is not None else color_data
                a_pos = 0 if alpha_buf is not None else used
                alpha = read_alpha(a_src, width, height, a_pos)
                if alpha is not None:
                    arr = bytearray(rgba)
                    arr[3::4] = alpha
                    rgba = bytes(arr)
            return encode_png(width, height, _ensure_visible(rgba, width, height))
        if not (flags & 0x10):  # no separate alpha plane: use transparent color
            arr = bytearray(rgba)
            if transparent is not None:
                mask = _color_key_match(rgba, width, height, transparent)
                arr[3::4] = mask.translate(_table(
                    lambda v: 0 if v == 0xFF else 255))
            else:
                arr[3::4] = b"\xff" * (width * height)
            return encode_png(width, height,
                              _ensure_visible(bytes(arr), width, height))
        # Alpha flag without RGBA: a separate alpha plane follows the pixels.
    elif mode == 16:
        rgba, used = read_32bpp(color_data, width, height)
        if not (flags & 0x10):
            return encode_png(width, height, _ensure_visible(rgba, width, height))
    elif mode == 0:
        # some MFA files store a 32-bit image with mode 0
        rgba, used = read_32bpp(color_data, width, height)
    else:
        return None

    n = width * height
    if flags & 0x10:  # alpha channel
        a_src = alpha_buf if alpha_buf is not None else color_data
        a_pos = 0 if alpha_buf is not None else used
        alpha = read_alpha(a_src, width, height, a_pos)
        if alpha is not None:
            arr = bytearray(rgba)
            arr[3::4] = alpha
            rgba = bytes(arr)
    elif transparent is not None:
        tr, tg, tb, _ta = transparent
        mask = _color_key_match(rgba, width, height, (tr, tg, tb))
        old = rgba[3::4]
        keyed = _and_bytes(mask.translate(_INVERT), old)
        arr = bytearray(rgba)
        arr[3::4] = keyed
        rgba = bytes(arr)

    return encode_png(width, height, _ensure_visible(rgba, width, height))


def _ensure_visible(rgba: bytes, width: int, height: int) -> bytes:
    """If every pixel is fully transparent but colour data is present, force
    opaque alpha.  Some Fusion banks leave alpha at 0 while still storing
    RGB, which made Scratch costumes load as blank invisible sprites.
    """
    if width <= 0 or height <= 0 or len(rgba) < width * height * 4:
        return rgba
    n = width * height
    if rgba[3] or rgba[n * 4 - 1]:    # the usual case: already visible
        return rgba
    alpha = rgba[3::4]
    if alpha.count(0) != n:            # something is already visible
        return rgba
    # No opaque pixel at all: only rescue it when there is colour to show.
    for offset in (0, 1, 2):
        if rgba[offset::4].count(0) != n:
            arr = bytearray(rgba)
            arr[3::4] = b"\xff" * n
            return bytes(arr)
    return rgba
