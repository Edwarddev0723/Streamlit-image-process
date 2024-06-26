import gc
import cv2
import torch
import numpy as np
from PIL import Image
from src import backbone
import matplotlib.pyplot as plt
from src.depthmap_generation import ModelHolder

def convert_to_i16(arr: np.array) -> np.array:
    # Single channel, 16 bit image. This loses some precision!
    # uint16 conversion uses round-down, therefore values should be [0; 2**16)
    max_val = 2 ** 16
    out = np.clip(arr * max_val + 0.0001, 0, max_val - 0.1)  # -0.1 from above is needed to avoid overflowing
    return out.astype("uint16")

def get_depthmap_new(original_img: Image, model_holder: ModelHolder) -> Image:

    try:
        # Convert single channel input (PIL) images to rgb
        if original_img.mode == 'I':
            original_img.point(lambda p: p * 0.0039063096, mode='RGB')
            original_img = original_img.convert('RGB')

        # Round up to a multiple of 32 to avoid potential issues
        net_width = (original_img.width + 31) // 32 * 32
        net_height = (original_img.height + 31) // 32 * 32

        # predict is being done here!
        raw_prediction, raw_prediction_invert = model_holder.get_raw_prediction(original_img, net_width, net_height)

        # output
        if abs(raw_prediction.max() - raw_prediction.min()) > np.finfo("float").eps:
            out = np.copy(raw_prediction)
            if raw_prediction_invert:
                out *= -1
            out = (out - out.min()) / (out.max() - out.min())  # normalize to [0; 1]
        else:
            out = np.zeros(raw_prediction.shape) # Regretfully, the depthmap is broken and will be replaced with a black image

        img_output = convert_to_i16(out)

    except Exception as e:
        raise e

    finally:
        # model_holder.offload()  # this unknown method makes errors on Colab merely but seems important, so just leave it here XDDD
        gc.collect()
        backbone.torch_gc()

    return Image.fromarray(img_output)

def display_resized(image: Image, new_height=0, new_width=0):
    original_width, original_height = image.size
    print(f"{original_width = }px\n{original_height = }px")

    if new_height or new_width:
        aspect_ratio = original_width / original_height
        new_height = new_height if new_height else int(new_width / aspect_ratio)
        new_width = new_width if new_width else int(new_height * aspect_ratio)
        resized_image = image.resize((new_width, new_height))
        display(resized_image)
    else:
        display(image)
    
def resized(image: Image, new_height=0, new_width=0):
    original_width, original_height = image.size
    print(f"{original_width = }px\n{original_height = }px")

    if new_height or new_width:
        aspect_ratio = original_width / original_height
        new_height = new_height if new_height else int(new_width / aspect_ratio)
        new_width = new_width if new_width else int(new_height * aspect_ratio)
        resized_image = image.resize((new_width, new_height))
        return resized_image
    else:
        return image

def binarized(image: Image, threshold=38, target_ratio=40) -> Image:
    """
    Binarize the input image.

    Args:
    image (Image): The input image.
    threshold (int): The threshold value for binarization. Defaults to 38 in Amazon demo figure.

    Returns:
    Image: The binarized image.
    """
    assert image.mode == 'I' or image.mode == 'I;16', image.mode
    assert isinstance(threshold, int) and 0 < threshold < 100
    assert isinstance(target_ratio, int) and 0 < target_ratio < 100

    image_array = np.array(image) // 256
    image_array = image_array.astype(np.uint8)

    black_ratio = 0
    while black_ratio < target_ratio / 100:
        _, bin_image = cv2.threshold(image_array, int(threshold/100*255), 255, cv2.THRESH_BINARY)
        black_pixels = np.count_nonzero(bin_image == 0)
        total_pixels = bin_image.size
        black_ratio = black_pixels / total_pixels
        threshold+=1

    return (binarized_image := Image.fromarray(bin_image))

def find_min_bounding_box_from_img(image: Image) -> np.array:
    """
    Find the minimum bounding box from the input image.

    Args:
    image (Image): The input image.

    Returns:
    np.array: The coordinates of the bounding box.
    """
    image_array = np.array(image.convert('L'))  # convert Image into np.array
    image_array = 255 - image_array  # invert black & white since cv2.findContours finds the white part only
    contours, _ = cv2.findContours(image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    return np.array([x, y, x + w, y + h])