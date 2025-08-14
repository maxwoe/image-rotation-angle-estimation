#!/usr/bin/env python3
"""
Clean rotation utilities that preserve image content without artificial borders
"""

import cv2
import numpy as np
import math
from PIL import Image


def rotate_image(img, angle, rotation_center=None, expand=False, border_mode=cv2.BORDER_CONSTANT, border_value=0):
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
        img, M, (wn, hn), borderMode=border_mode, borderValue=border_value)

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


def rotate_image_crop_max_area(image, angle):
    """
    Rotate image and crop to maximum area (clean implementation)
    
    Args:
        image: OpenCV image (numpy array)
        angle: Rotation angle in degrees
        
    Returns:
        numpy array: Rotated and cropped image
    """
    h, w = image.shape[:2]
    
    # Rotate image with expansion to fit all content
    rotated, rotation_matrix = rotate_image(image, angle, expand=True)
    
    # Calculate optimal crop dimensions using the improved formula
    wr, hr = largest_rotated_rect(w, h, math.radians(angle))
    
    # Get dimensions of rotated image
    h_rot, w_rot = rotated.shape[:2]
    
    # Calculate crop coordinates (center crop)
    y1 = h_rot//2 - int(hr/2)
    y2 = y1 + int(hr)
    x1 = w_rot//2 - int(wr/2)
    x2 = x1 + int(wr)
    
    # Crop the rotated image to maximum content area
    cropped = rotated[y1:y2, x1:x2]
    
    return cropped


def rotate_preserve_content(image_path, angle):
    """
    Rotate image preserving content, no artificial borders
    Returns the rotated image without resizing (let transforms handle that)
    
    Args:
        image_path: Path to image file
        angle: Rotation angle in degrees
    
    Returns:
        PIL Image: Rotated and cropped image (original content preserved)
    """
    # Load image with OpenCV
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Rotate and crop to preserve maximum content area
    image_rotated_cropped = rotate_image_crop_max_area(image, angle)
    
    # Convert BGR to RGB
    image_rotated_cropped = cv2.cvtColor(image_rotated_cropped, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL (no resizing - let transforms handle it)
    final_image = Image.fromarray(image_rotated_cropped)
    
    return final_image