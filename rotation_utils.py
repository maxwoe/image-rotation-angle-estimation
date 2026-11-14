#!/usr/bin/env python3
"""
Rotation utilities that preserve image content without artificial borders.

Two groups of functions live here.

Paper protocol (unchanged from the published code, so that the paper's numbers
stay reproducible): ``rotate_image``, ``largest_rotated_rect``,
``rotate_image_crop_max_area`` and ``rotate_preserve_content`` with default
arguments decode the source, rotate it on an expanded canvas with bilinear
resampling, centre-crop to the largest inscribed axis-aligned rectangle and
never re-encode the result.

Trace removal: the paper protocol leaves two traces in every view
that carry the applied angle independently of the image content. The source
photo's 8x8 JPEG block grid rotates with the content, so its orientation gives
the angle modulo 90 degrees with sub-degree precision, and the rotation
resampler leaves a periodic residue aligned with the source pixel grid. A model
trained on such views learns to read the fine angle from these traces: one JPEG
save of the test views drops the published COCO model from 98% to 30% within
10 degrees, with predictions snapping to the nearest axis (reported by ORTUS AI,
September 2026, and reproduced here). ``TraceRemoval`` removes the traces from
a rotated view:

1. the resampler is drawn at random per view,
2. an isotropic Gaussian blur attenuates the 8 px grid period (the residual
   amplitude of an 8 px periodic pattern is exp(-2 pi^2 sigma^2 / 64):
   sigma 2.0 -> 0.29, 2.5 -> 0.15, 3.0 -> 0.06, 4.0 -> 0.007), and
3. a JPEG re-encode after rotation and crop, at a random quality, adds the
   frame-aligned grid a camera would produce (which carries no angle
   information) and masks any residue.

Rotation mode "crop_fixed" is an angle-independent crop (a rectangle with the
source aspect ratio inscribed in the source's inscribed circle), offered as a
control for the field-of-view coupling of the max-area crop.
"""

import io
import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


RESAMPLERS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}


def rotate_image(img, angle, rotation_center=None, expand=False, border_mode=cv2.BORDER_CONSTANT, border_value=0,
                 interpolation=cv2.INTER_LINEAR):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    h, w = img.shape[:2]
    if rotation_center is None:
        rotation_center = (w/2, h/2)  # img center

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)

    if expand:
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])
        # find the new width and height bounds
        wn = int(h * abs_sin + w * abs_cos)
        hn = int(h * abs_cos + w * abs_sin)
        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        M[0, 2] += wn/2 - rotation_center[0]
        M[1, 2] += hn/2 - rotation_center[1]
    else:
        wn, hn = w, h

    rotated = cv2.warpAffine(
        img, M, (wn, hn), flags=interpolation, borderMode=border_mode, borderValue=border_value)

    return rotated, M


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr, hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr, hr


def fixed_crop_rect(w, h):
    """
    Angle-independent crop: a rectangle with the source aspect ratio whose
    diagonal is min(w, h) - 2, so it lies inside the circle inscribed in the
    source at every rotation angle. Unlike ``largest_rotated_rect`` its size
    and field of view do not depend on the angle.
    """
    d = min(w, h) - 2
    diag = math.hypot(w, h)
    return d * w / diag, d * h / diag


def _center_crop(rotated, wr, hr):
    h_rot, w_rot = rotated.shape[:2]
    y1 = h_rot//2 - int(hr/2)
    y2 = y1 + int(hr)
    x1 = w_rot//2 - int(wr/2)
    x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2]


def rotate_image_crop_max_area(image, angle, interpolation=cv2.INTER_LINEAR):
    """
    Rotate image and crop to maximum area (paper protocol)

    Args:
        image: OpenCV image (numpy array)
        angle: Rotation angle in degrees
        interpolation: OpenCV resampling flag (default bilinear, as in the paper)

    Returns:
        numpy array: Rotated and cropped image
    """
    h, w = image.shape[:2]

    # Rotate image with expansion to fit all content
    rotated, rotation_matrix = rotate_image(image, angle, expand=True, interpolation=interpolation)

    # Calculate optimal crop dimensions using the improved formula
    wr, hr = largest_rotated_rect(w, h, math.radians(angle))

    # Crop the rotated image to maximum content area (center crop)
    return _center_crop(rotated, wr, hr)


def rotate_image_crop_fixed(image, angle, interpolation=cv2.INTER_LINEAR):
    """Rotate image and crop to the angle-independent rectangle of ``fixed_crop_rect``."""
    h, w = image.shape[:2]
    rotated, _ = rotate_image(image, angle, expand=True, interpolation=interpolation)
    wr, hr = fixed_crop_rect(w, h)
    return _center_crop(rotated, wr, hr)


def jpeg_roundtrip(image_bgr, quality):
    """
    Encode a BGR array as JPEG at the given quality (Pillow, 4:2:0 chroma
    subsampling, as a camera or photo pipeline would) and decode it again.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    rgb = np.array(Image.open(buf).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _parse_pair(text, cast):
    lo, hi = (cast(v) for v in str(text).split(","))
    return lo, hi


@dataclass
class TraceRemoval:
    """
    Per-view processing that keeps processing traces from carrying the angle.
    The defaults are the trace-free protocol of the paper: every rotated and
    cropped view is re-encoded once as JPEG at a quality drawn from 40 to 90,
    and the rotation resampler is drawn at random; no blur, no downscale.

    blur_sigma:   (lo, hi) of an optional Gaussian blur, sigma in pixels of the
                  rotated view, drawn uniformly; (0, 0) disables the blur.
    jpeg_quality: (lo, hi) of the JPEG quality drawn uniformly for the re-encode.
    jpeg_prob:    probability of re-encoding a view; below 1 the rest stay
                  lossless.
    resamplers:   names from RESAMPLERS, one drawn per view for the rotation.
    """
    blur_sigma: Tuple[float, float] = (0.0, 0.0)
    jpeg_quality: Tuple[int, int] = (40, 90)
    jpeg_prob: float = 1.0
    resamplers: Sequence[str] = ("linear", "cubic", "lanczos4")
    scale: Tuple[float, float] = (1.0, 1.0)   # random area-downscale factor of the view, (1, 1) = off

    @classmethod
    def from_env(cls) -> "TraceRemoval":
        """Build from REMOVE_TRACES_* environment variables (set by train.py)."""
        cfg = cls()
        if os.environ.get("REMOVE_TRACES_SCALE"):
            cfg.scale = _parse_pair(os.environ["REMOVE_TRACES_SCALE"], float)
        if os.environ.get("REMOVE_TRACES_BLUR"):
            cfg.blur_sigma = _parse_pair(os.environ["REMOVE_TRACES_BLUR"], float)
        if os.environ.get("REMOVE_TRACES_JPEG"):
            cfg.jpeg_quality = _parse_pair(os.environ["REMOVE_TRACES_JPEG"], int)
        if os.environ.get("REMOVE_TRACES_JPEG_PROB"):
            cfg.jpeg_prob = float(os.environ["REMOVE_TRACES_JPEG_PROB"])
        if os.environ.get("REMOVE_TRACES_RESAMPLERS"):
            cfg.resamplers = tuple(s.strip() for s in os.environ["REMOVE_TRACES_RESAMPLERS"].split(",") if s.strip())
        for name in cfg.resamplers:
            if name not in RESAMPLERS:
                raise ValueError(f"Unknown resampler '{name}'. Choose from {sorted(RESAMPLERS)}")
        return cfg

    def pick_resampler(self, rng) -> int:
        return RESAMPLERS[rng.choice(list(self.resamplers))]

    def apply(self, image_bgr, rng):
        """Apply downscale, blur and (with probability jpeg_prob) a JPEG re-encode to a rotated, cropped view."""
        s_lo, s_hi = self.scale
        if s_hi < 1.0 or s_lo < 1.0:
            s = rng.uniform(s_lo, s_hi)
            if s < 0.999:
                h, w = image_bgr.shape[:2]
                image_bgr = cv2.resize(image_bgr, (max(8, int(round(w * s))), max(8, int(round(h * s)))),
                                       interpolation=cv2.INTER_AREA)
        lo, hi = self.blur_sigma
        if hi > 0:
            sigma = rng.uniform(lo, hi)
            if sigma > 0:
                image_bgr = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
        if self.jpeg_prob > 0 and rng.random() < self.jpeg_prob:
            q_lo, q_hi = self.jpeg_quality
            image_bgr = jpeg_roundtrip(image_bgr, rng.randint(int(q_lo), int(q_hi)))
        return image_bgr

    def describe(self) -> str:
        return (f"scale {self.scale[0]}-{self.scale[1]}, blur sigma {self.blur_sigma[0]}-{self.blur_sigma[1]} px, "
                f"JPEG q{self.jpeg_quality[0]}-{self.jpeg_quality[1]} with p={self.jpeg_prob}, "
                f"resamplers {','.join(self.resamplers)}")


CIRCLE_FILL_BGR = (104, 116, 124)   # ORTUS AI's letterbox fill (RGB 124, 116, 104), carries no information


def rotate_image_circle(image, angle, interpolation=cv2.INTER_LINEAR):
    """
    Rotate and keep only the circle inscribed in the source (Follmann and
    Boettger's circular crop): the centre square of side min(w, h) of the
    rotated canvas with everything outside the circle of radius min(w, h)/2
    set to a constant fill. The circle is covered by content at every angle
    and its outline is rotation-invariant, so neither size, shape nor border
    carries the angle; the price is the lost corners.
    """
    h, w = image.shape[:2]
    d = min(h, w)
    rotated, _ = rotate_image(image, angle, expand=True, interpolation=interpolation)
    square = np.ascontiguousarray(_center_crop(rotated, d, d))
    mask = np.zeros(square.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (square.shape[1] // 2, square.shape[0] // 2), d // 2 - 1, 255, -1)
    out = np.empty_like(square)
    out[:] = np.array(CIRCLE_FILL_BGR, dtype=square.dtype)
    out[mask > 0] = square[mask > 0]
    return out


def rotate_array_preserve_content(image, angle, mode="crop", interpolation=cv2.INTER_LINEAR):
    """Rotate a BGR array according to ``mode`` ("crop", "crop_fixed", "circle" or "pad")."""
    if mode == "pad":
        # Rotate with expansion, keep black corners (no crop)
        rotated, _ = rotate_image(image, angle, expand=True, interpolation=interpolation)
        return rotated
    if mode == "crop":
        # Rotate and crop to maximum content area
        return rotate_image_crop_max_area(image, angle, interpolation=interpolation)
    if mode == "crop_fixed":
        return rotate_image_crop_fixed(image, angle, interpolation=interpolation)
    if mode == "circle":
        return rotate_image_circle(image, angle, interpolation=interpolation)
    raise ValueError(f"Unknown rotation mode: {mode}. Use 'crop', 'crop_fixed', 'circle' or 'pad'")


ROTATION_MODES = ("crop", "crop_fixed", "circle", "pad")


def rotate_preserve_content(image_path, angle, mode="crop", trace_removal: Optional[TraceRemoval] = None,
                            rng=None, blur_sigma: float = 0.0, jpeg_quality: int = 0):
    """
    Rotate image preserving content
    Returns the rotated image without resizing (let transforms handle that)

    Args:
        image_path: Path to image file
        angle: Rotation angle in degrees
        mode: "crop" = center-crop to largest inscribed axis-aligned rectangle
              (no artificial borders, paper protocol); "crop_fixed" = angle-
              independent crop; "pad" = keep the full rotated image with black
              corner padding (control experiment for crop-boundary cues)
        trace_removal: optional TraceRemoval config; when given, the resampler
              is drawn at random and the view has its traces removed (blur + JPEG)
        rng: random.Random instance used for the trace-removal draws (default:
              the global random module)
        blur_sigma: if > 0, blur the rotated view with this Gaussian sigma
              (test-time trace removal for models trained with it; applied
              before any recompression)
        jpeg_quality: if > 0, re-encode the final view once as JPEG at this
              quality (test-time recompression control; applied last)

    Returns:
        PIL Image: Rotated image (original content preserved)
    """
    # Load image with OpenCV
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    interpolation = cv2.INTER_LINEAR
    if trace_removal is not None:
        rng = rng if rng is not None else random
        interpolation = trace_removal.pick_resampler(rng)

    image_rotated = rotate_array_preserve_content(image, angle, mode=mode, interpolation=interpolation)

    if trace_removal is not None:
        image_rotated = trace_removal.apply(image_rotated, rng)
    if blur_sigma and blur_sigma > 0:
        image_rotated = cv2.GaussianBlur(image_rotated, (0, 0), float(blur_sigma))
    if jpeg_quality and jpeg_quality > 0:
        image_rotated = jpeg_roundtrip(image_rotated, jpeg_quality)

    # Convert BGR to RGB
    image_rotated = cv2.cvtColor(image_rotated, cv2.COLOR_BGR2RGB)

    # Convert to PIL (no resizing - let transforms handle it)
    final_image = Image.fromarray(image_rotated)

    return final_image
