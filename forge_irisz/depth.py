# Iris-Z support for Forge Neo
# Image to Depth's sizes and output, and Depth to Image's input, as the SDK's image_to_depth and depth_to_image
# (iris_z/pipeline.py and iris_z/depth.py).
#
# The depth grid decodes through the FLUX.2 VAE like an image whose three channels hold the same value: the
# relative log depth of the picture, its 2nd to 98th percentile spread over [-1, 1], far = +1. Iris-Z-small has
# no sky ceiling (sky_ceiling 1.0), so that value is the depth as it is. The map is written near = bright, the
# convention depth ControlNets read.

import math

from PIL import Image

# under half a megapixel the SDK works at 512, above at 1024
SMALL_IMAGE_PIXELS = 500_000


def choose_resolution(source_width, source_height, target_side, multiple=16, area_tolerance=0.2):
    """About target_side^2 pixels, same aspect ratio, multiples of 16, never larger than the source."""
    source_aspect = source_width / source_height
    target_area = min(target_side * target_side, source_width * source_height)
    best = None
    for height in range(multiple, source_height + 1, multiple):
        for width in range(multiple, source_width + 1, multiple):
            area = width * height
            if abs(area - target_area) > area_tolerance * target_area:
                continue
            aspect_error = abs(math.log((width / height) / source_aspect))
            key = (round(aspect_error, 6), abs(area - target_area))
            if best is None or key < best[0]:
                best = (key, width, height)
    if best is None:
        raise ValueError(f"no size near {target_side}^2 fits {source_width}x{source_height}")
    return best[1], best[2]


def work_size(width, height):
    side = 512 if width * height < SMALL_IMAGE_PIXELS else 1024
    return choose_resolution(width, height, side)


def enlarged_size(width, height, target_side=1024):
    """Like choose_resolution, but a smaller source is enlarged: Depth to Image always paints at about target_side^2."""
    area = width * height
    if area < target_side * target_side:
        factor = (target_side * target_side / area) ** 0.5
        width, height = round(width * factor), round(height * factor)
    return choose_resolution(width, height, target_side)


def paint_input(depth_image: Image.Image) -> Image.Image:
    """A near = bright depth map as the picture the VAE encodes into the given depth grid.

    The grid holds the depth encoded as the SDK's encode_log_depth: far = +1, the 2nd to 98th percentile spread over
    [-1, 1], the same value in the three channels. A map from Image to Depth is that encoding inverted (depth_map),
    so inverting it gives the encoding back; any other near = bright map is read the same way, its range stretched
    like the SDK stretches it.
    """
    import numpy as np

    size = enlarged_size(*depth_image.size)
    grey = depth_image.convert("L").resize(size, Image.Resampling.BILINEAR)
    encoded = 1.0 - np.asarray(grey, dtype=np.float64) / 127.5  # near -1 .. far +1
    low, high = np.percentile(encoded, [2, 98])
    encoded = np.clip(2.0 * (encoded - low) / max(high - low, 1e-6) - 1.0, -1.0, 1.0)
    value = np.round(127.5 * (encoded + 1.0)).astype(np.uint8)
    return Image.fromarray(value, mode="L").convert("RGB")


def depth_map(decoded: Image.Image, size) -> Image.Image:
    """The decoded depth grid as a grey map at size (width, height), near = bright."""
    import numpy as np

    value = np.asarray(decoded.convert("RGB"), dtype=np.float32).mean(axis=2)  # 0 near .. 255 far
    near_bright = np.clip(255.0 - value, 0.0, 255.0)
    image = Image.fromarray(near_bright.round().astype(np.uint8), mode="L")
    if image.size != tuple(size):
        image = image.resize(tuple(size), Image.Resampling.BILINEAR)
    return image
