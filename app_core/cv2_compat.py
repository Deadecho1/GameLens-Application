from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image


def prepare_cv2_for_ultralytics() -> None:
    """Patch partially-broken cv2 installs so Ultralytics can import and run.

    Some environments can end up with a namespace-only ``cv2`` package where core
    symbols (e.g. ``IMREAD_COLOR``) are missing. Ultralytics references these at
    import-time, which otherwise crashes the boss classifier startup.

    This function applies minimal compatibility shims **only for missing symbols**.
    A healthy OpenCV installation remains untouched.
    """

    try:
        import cv2
    except Exception:
        return

    # Common constants used by Ultralytics/OpenCV call sites.
    constants: dict[str, int] = {
        "IMREAD_COLOR": 1,
        "IMREAD_GRAYSCALE": 0,
        "IMREAD_UNCHANGED": -1,
        "COLOR_BGR2RGB": 4,
        "COLOR_RGB2BGR": 4,
        "COLOR_BGR2GRAY": 6,
        "COLOR_RGB2GRAY": 7,
        "INTER_NEAREST": 0,
        "INTER_LINEAR": 1,
        "INTER_AREA": 3,
        "BORDER_CONSTANT": 0,
        "FONT_HERSHEY_SIMPLEX": 0,
        "LINE_AA": 16,
    }
    for key, value in constants.items():
        if not hasattr(cv2, key):
            setattr(cv2, key, value)

    # GUI/utility stubs (safe in headless mode).
    if not hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads = lambda *_a, **_k: None  # type: ignore[attr-defined]
    if not hasattr(cv2, "imshow"):
        cv2.imshow = lambda *_a, **_k: None  # type: ignore[attr-defined]
    if not hasattr(cv2, "waitKey"):
        cv2.waitKey = lambda *_a, **_k: -1  # type: ignore[attr-defined]
    if not hasattr(cv2, "destroyAllWindows"):
        cv2.destroyAllWindows = lambda *_a, **_k: None  # type: ignore[attr-defined]

    if not hasattr(cv2, "getTextSize"):
        cv2.getTextSize = lambda *_a, **_k: ((0, 0), 0)  # type: ignore[attr-defined]
    if not hasattr(cv2, "putText"):
        cv2.putText = lambda img, *_a, **_k: img  # type: ignore[attr-defined]
    if not hasattr(cv2, "rectangle"):
        cv2.rectangle = lambda img, *_a, **_k: img  # type: ignore[attr-defined]

    if not hasattr(cv2, "copyMakeBorder"):

        def _copy_make_border(
            img: np.ndarray,
            top: int,
            bottom: int,
            left: int,
            right: int,
            _border_type: int,
            value: Any = 0,
        ) -> np.ndarray:
            if img.ndim == 2:
                pad = ((top, bottom), (left, right))
            else:
                pad = ((top, bottom), (left, right), (0, 0))

            if isinstance(value, tuple):
                # np.pad(constant_values=...) expects per-axis values.
                # For color tuples we can safely use the first scalar fallback.
                fill = value[0] if value else 0
            else:
                fill = value
            return np.pad(img, pad, mode="constant", constant_values=fill)

        cv2.copyMakeBorder = _copy_make_border  # type: ignore[attr-defined]

    if not hasattr(cv2, "resize"):

        def _resize(
            img: np.ndarray, dsize: tuple[int, int], interpolation: int = 1
        ) -> np.ndarray:
            pil = Image.fromarray(img)
            resample = (
                Image.Resampling.BILINEAR
                if interpolation == getattr(cv2, "INTER_LINEAR", 1)
                else Image.Resampling.NEAREST
            )
            return np.asarray(pil.resize(dsize, resample))

        cv2.resize = _resize  # type: ignore[attr-defined]

    if not hasattr(cv2, "cvtColor"):

        def _cvt_color(img: np.ndarray, code: int) -> np.ndarray:
            if code in (
                getattr(cv2, "COLOR_BGR2RGB", 4),
                getattr(cv2, "COLOR_RGB2BGR", 4),
            ):
                if img.ndim == 3 and img.shape[2] >= 3:
                    return img[..., ::-1]
                return img
            if code in (
                getattr(cv2, "COLOR_BGR2GRAY", 6),
                getattr(cv2, "COLOR_RGB2GRAY", 7),
            ):
                if img.ndim == 2:
                    return img
                return np.asarray(Image.fromarray(img).convert("L"))
            return img

        cv2.cvtColor = _cvt_color  # type: ignore[attr-defined]

    if not hasattr(cv2, "imdecode"):

        def _imdecode(buf: Any, flags: int = 1) -> np.ndarray | None:
            try:
                data = buf.tobytes() if hasattr(buf, "tobytes") else bytes(buf)
                with Image.open(io.BytesIO(data)) as im:
                    if flags == getattr(cv2, "IMREAD_GRAYSCALE", 0):
                        return np.asarray(im.convert("L"))
                    # OpenCV convention is BGR.
                    return np.asarray(im.convert("RGB"))[..., ::-1]
            except Exception:
                return None

        cv2.imdecode = _imdecode  # type: ignore[attr-defined]

    if not hasattr(cv2, "imdecodemulti"):
        cv2.imdecodemulti = lambda *_a, **_k: (False, [])  # type: ignore[attr-defined]

    if not hasattr(cv2, "imencode"):

        def _imencode(ext: str, img: np.ndarray, _params: list[int] | None = None):
            suffix = (ext or ".png").lstrip(".").lower()
            pil_fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}.get(
                suffix, "PNG"
            )
            pil_img = Image.fromarray(
                img[..., ::-1] if img.ndim == 3 and img.shape[2] >= 3 else img
            )
            out = io.BytesIO()
            pil_img.save(out, format=pil_fmt)
            arr = np.frombuffer(out.getvalue(), dtype=np.uint8)
            out.close()
            return True, arr

        cv2.imencode = _imencode  # type: ignore[attr-defined]

    if not hasattr(cv2, "__getattr__"):

        def _fallback_getattr(name: str):
            if name.startswith(
                (
                    "IMREAD_",
                    "COLOR_",
                    "INTER_",
                    "BORDER_",
                    "FONT_",
                    "LINE_",
                    "CAP_PROP_",
                    "WINDOW_",
                )
            ):
                return 0
            raise AttributeError(f"module 'cv2' has no attribute {name!r}")

        cv2.__getattr__ = _fallback_getattr  # type: ignore[attr-defined]
