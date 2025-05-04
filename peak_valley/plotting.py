from __future__ import annotations
import io, base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

__all__ = ["fig_to_png", "thumb64"]


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0.15)
    buf.seek(0)
    return buf.getvalue()


def thumb64(png_bytes: bytes) -> str:
    im = Image.open(io.BytesIO(png_bytes)).resize((150, 100))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
