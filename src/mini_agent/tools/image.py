"""图片嗅探和缩放。Read 读到 jpg/png/gif/webp/bmp 时用这里，不另设工具。

对齐 Pi 的 detectSupportedImageMimeType + processImage：魔数判断，能直接
内联的原样保留，bmp 等先转成 png，边长超过 2000 或 base64 超过约 4.5MB 就缩小。
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

_PNG = bytes((0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A))
MAX_EDGE = 2000
MAX_BASE64_CHARS = int(4.5 * 1024 * 1024)
_SNIFF_BYTES = 4100


@dataclass
class ProcessedImage:
    ok: bool
    data: str = ""
    mime_type: str = ""
    hints: list[str] | None = None
    message: str = ""


def detect_image_mime(buffer: bytes) -> str | None:
    """认 jpg/png/gif/webp/bmp。动画 PNG 和 JPEG-LS 不当图片。"""
    if buffer.startswith(b"\xff\xd8\xff"):
        return None if len(buffer) > 3 and buffer[3] == 0xF7 else "image/jpeg"
    if buffer.startswith(_PNG):
        return "image/png" if _is_png(buffer) and not _is_animated_png(buffer) else None
    if buffer.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if buffer.startswith(b"RIFF") and buffer[8:12] == b"WEBP":
        return "image/webp"
    if buffer.startswith(b"BM") and _is_bmp(buffer):
        return "image/bmp"
    return None


def sniff_image_mime(path) -> str | None:
    with open(path, "rb") as fh:
        return detect_image_mime(fh.read(_SNIFF_BYTES))


def process_image(data: bytes, mime_type: str) -> ProcessedImage:
    """返回可放进工具结果的 base64。转不了或缩不进上限时 ok=False。"""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return ProcessedImage(
            ok=False,
            message="[Image omitted: Pillow 未安装，无法处理图片。]",
        )

    original_mime = mime_type
    if mime_type in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
        try:
            with Image.open(io.BytesIO(data)) as probe:
                probe.load()
                width, height = probe.size
        except OSError:
            return ProcessedImage(
                ok=False,
                message="[Image omitted: could not be converted to a supported inline image format.]",
            )
        if width <= MAX_EDGE and height <= MAX_EDGE and _b64_len(data) <= MAX_BASE64_CHARS:
            return ProcessedImage(
                ok=True,
                data=base64.b64encode(data).decode("ascii"),
                mime_type=mime_type,
                hints=[],
            )

    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img) or img
            if getattr(img, "is_animated", False):
                img.seek(0)
            img.load()
            original_size = img.size
            working = img.copy()
    except OSError:
        return ProcessedImage(
            ok=False,
            message="[Image omitted: could not be converted to a supported inline image format.]",
        )

    working.thumbnail((MAX_EDGE, MAX_EDGE))
    # bmp 等不能直接内联的格式先转成 png。转完仍放得下就用 png，
    # 跟 Pi 的 normalizeImage 一样；放不下再走缩小和 jpeg。
    if original_mime not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
        png_bytes = _save(_for_png(working), "PNG")
        if _b64_len(png_bytes) <= MAX_BASE64_CHARS:
            hints = [f"[Image converted from {original_mime} to image/png.]"]
            note = _dimension_note(original_size, working.size)
            if note:
                hints.append(note)
            return ProcessedImage(
                ok=True,
                data=base64.b64encode(png_bytes).decode("ascii"),
                mime_type="image/png",
                hints=hints,
            )
    encoded = _encode_under_limit(working)
    if encoded is None:
        return ProcessedImage(
            ok=False,
            message="[Image omitted: could not be resized below the inline image size limit.]",
        )
    raw, out_mime, shown = encoded
    hints: list[str] = []
    if original_mime != out_mime:
        hints.append(f"[Image converted from {original_mime} to {out_mime}.]")
    note = _dimension_note(original_size, shown)
    if note:
        hints.append(note)
    return ProcessedImage(
        ok=True,
        data=base64.b64encode(raw).decode("ascii"),
        mime_type=out_mime,
        hints=hints,
    )


def _for_png(img):
    if img.mode in {"1", "L", "P", "RGB", "RGBA", "LA"}:
        return img
    return img.convert("RGBA" if "A" in img.getbands() else "RGB")


def _dimension_note(original: tuple[int, int], shown: tuple[int, int]) -> str | None:
    if original == shown or not shown[0]:
        return None
    scale = original[0] / shown[0]
    return (
        f"[Image: original {original[0]}x{original[1]}, "
        f"displayed at {shown[0]}x{shown[1]}. "
        f"Multiply coordinates by {scale:.2f} to map to original image.]"
    )


def _encode_under_limit(img) -> tuple[bytes, str, tuple[int, int]] | None:
    """先按当前尺寸编码，仍超限就缩小再试。返回字节、mime、最终边长。"""
    from PIL import Image

    current = img
    for _ in range(12):
        has_alpha = current.mode in {"RGBA", "LA"} or (
            current.mode == "P" and "transparency" in current.info
        )
        if has_alpha:
            encoded = _save(current.convert("RGBA"), "PNG")
            if _b64_len(encoded) <= MAX_BASE64_CHARS:
                return encoded, "image/png", current.size
        else:
            rgb = current.convert("RGB")
            for quality in (80, 60, 40, 25):
                encoded = _save(rgb, "JPEG", quality=quality)
                if _b64_len(encoded) <= MAX_BASE64_CHARS:
                    return encoded, "image/jpeg", current.size
        if current.size[0] <= 1 and current.size[1] <= 1:
            return None
        nxt = current.copy()
        nxt.thumbnail(
            (max(1, current.size[0] * 3 // 4), max(1, current.size[1] * 3 // 4)),
            Image.Resampling.LANCZOS,
        )
        if nxt.size == current.size:
            return None
        current = nxt
    return None


def _save(img, fmt: str, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format=fmt, quality=quality)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def _b64_len(raw: bytes) -> int:
    return (len(raw) + 2) // 3 * 4


def _u32be(buf: bytes, offset: int) -> int:
    return int.from_bytes(buf[offset : offset + 4], "big")


def _u32le(buf: bytes, offset: int) -> int:
    return int.from_bytes(buf[offset : offset + 4], "little")


def _u16le(buf: bytes, offset: int) -> int:
    return int.from_bytes(buf[offset : offset + 2], "little")


def _is_png(buf: bytes) -> bool:
    return len(buf) >= 16 and _u32be(buf, len(_PNG)) == 13 and buf[12:16] == b"IHDR"


def _is_animated_png(buf: bytes) -> bool:
    offset = len(_PNG)
    while offset + 8 <= len(buf):
        length = _u32be(buf, offset)
        kind = buf[offset + 4 : offset + 8]
        if kind == b"acTL":
            return True
        if kind == b"IDAT":
            return False
        nxt = offset + 8 + length + 4
        if nxt <= offset or nxt > len(buf):
            return False
        offset = nxt
    return False


def _is_bmp(buf: bytes) -> bool:
    if len(buf) < 26:
        return False
    declared = _u32le(buf, 2)
    pixel = _u32le(buf, 10)
    dib = _u32le(buf, 14)
    if declared != 0 and declared < 26:
        return False
    if pixel < 14 + dib:
        return False
    if declared != 0 and pixel >= declared:
        return False
    if dib == 12:
        planes, bpp = _u16le(buf, 22), _u16le(buf, 24)
    elif 40 <= dib <= 124 and len(buf) >= 30:
        planes, bpp = _u16le(buf, 26), _u16le(buf, 28)
    else:
        return False
    return planes == 1 and bpp in {1, 4, 8, 16, 24, 32}
