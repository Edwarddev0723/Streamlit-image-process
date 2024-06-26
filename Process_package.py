import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image, ImageFilter

from IPython.display import display

def get_mask(path: str) -> np.array: #just for testing #just for testing #just for testing
    img = Image.open(path)
    img_arr = np.array(img)[:, :, 3] #get alpha only
    img_arr = (img_arr > .5).astype(bool)
    return img_arr

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.5])
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def size_optimized(image: Image, largest_inner_box: np.array) -> Image:
    assert len(largest_inner_box) == 4
    w = largest_inner_box[2] - largest_inner_box[0]
    h = largest_inner_box[3] - largest_inner_box[1]

    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    new_height = h if min(w,h)==h else int(w / aspect_ratio)
    new_width = w if min(w,h)==w else int(h * aspect_ratio)

    return (resized_image := image.resize((new_width, new_height)))

def find_best_position(largest_inner_box: np.array, ad: Image) -> tuple[int, int]:
    assert len(largest_inner_box) == 4
    box_w = largest_inner_box[2] - largest_inner_box[0]
    box_h = largest_inner_box[3] - largest_inner_box[1]
    ad_w, ad_h = ad.size

    if box_w == ad_w:
        x = largest_inner_box[0]
        y = (largest_inner_box[3] + largest_inner_box[1]) // 2 - ad_h // 2
    elif box_h == ad_h:
        x = (largest_inner_box[2] + largest_inner_box[0]) // 2 - ad_w // 2
        y = largest_inner_box[1]
    else:
        raise ValueError("Largest inner box & ad must have a same side at least!")


    return (coordinate := (max(x,0), max(y,0)))

def display_resized(image: Image, new_height=0, new_width=0):
    original_width, original_height = image.size
    print(f"width = {original_width}px\nheight = {original_height}px")

    if new_height or new_width:
        aspect_ratio = original_width / original_height
        new_height = new_height if new_height else int(new_width / aspect_ratio)
        new_width = new_width if new_width else int(new_height * aspect_ratio)
        resized_image = image.resize((new_width, new_height))
        display(resized_image)
    else:
        display(image)

#@title blending modes
def apply_alpha(func): # a decorator to adjusting opacity
    def func_applied_alpha(top: np.array, bottom: np.array, alpha=1):
        assert 0 <= alpha <= 1, "alpha must be between 0 and 1 !"
        res = func(top, bottom)
        res = res * alpha + bottom * (1 - alpha)
        return res
    return func_applied_alpha

@apply_alpha
def normal(top: np.array, bottom: np.array) -> np.array: # 正常
    res = top
    res = res.astype(np.uint8)
    return res

# this blending mode isn't decorated by "apply_alpah" since its alpha is special
def dissolve(top: np.array, bottom: np.array, alpha=1) -> np.array: # 溶解
    sample = np.random.uniform(size=top.shape) > (1 - alpha)
    res = top * sample + bottom * (1 - sample)
    return res.astype(np.uint8)

@apply_alpha
def multiply(top: np.array, bottom: np.array) -> np.array: # 色彩增值
    top, bottom = top / 255.0, bottom / 255.0
    res = top * bottom
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def screen(top: np.array, bottom: np.array) -> np.array: # 濾色
    top, bottom = top / 255.0, bottom / 255.0
    res = 1 - (1 - top) * (1 - bottom)
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def overlay(top: np.array, bottom: np.array) -> np.array: # 覆蓋
    top, bottom = top / 255.0, bottom / 255.0
    res = np.where(bottom < 0.5,
             2 * top * bottom,
             1 - 2 * (1 - top) * (1 - bottom))
    res = (255 * res).astype(np.uint8)
    return res

@apply_alpha
def soft_light(top: np.array, bottom: np.array) -> np.array: # 柔光
    top, bottom = top / 255.0, bottom / 255.0
    res = np.where(top < 0.5,
             (2 * top - 1) * (bottom - bottom**2) + bottom,
             (2 * top - 1) * (np.sqrt(bottom) - bottom) + bottom)
    res = (255 * res).astype(np.uint8)
    return res

@apply_alpha
def hard_light(top: np.array, bottom: np.array) -> np.array: # 實光
    top, bottom = top / 255.0, bottom / 255.0
    res = np.where(top < 0.5,
             2 * top * bottom,
             1 - 2 * (1 - top) * (1 - bottom))
    res = (255 * res).astype(np.uint8)
    return res

@apply_alpha
def color_dodge(top: np.array, bottom: np.array) -> np.array: # 加亮顏色
    top, bottom = top / 255.0, bottom / 255.0
    res = bottom / (1.0 - top + 0.001)  # prevent ZeroDivisionError
    res = np.clip(res, 0, 1)  # Ensures values are within [0, 1]
    res = (255 * res).astype(np.uint8)
    return res

@apply_alpha
def linear_dodge(top: np.array, bottom: np.array) -> np.array: # 線性加亮
    top, bottom = top / 255.0, bottom / 255.0
    res = np.clip(top + bottom, 0, 1)  # clips sum to [0, 1]
    res = (255 * res).astype(np.uint8)
    return res

@apply_alpha
def color_burn(top: np.array, bottom: np.array) -> np.array: # 加深顏色/顏色加深
    top, bottom = top / 255.0, bottom / 255.0
    res = 1.0 - (1.0 - bottom) / (top + 0.001)  # prevent ZeroDivisionError
    res = np.clip(res, 0, 1)  # Ensures values are within [0, 1]
    res = (255 * res).astype(np.uint8)
    return res

@apply_alpha
def linear_burn(top: np.array, bottom: np.array) -> np.array: # 線性加深
    top, bottom = top / 255.0, bottom / 255.0
    res = np.clip(top + bottom - 1.0, 0, 1)  # Sums, adjusts, and clips values to [0, 1]
    res = (255 * res).astype(np.uint8)
    return res

@apply_alpha
def vivid_light(top: np.array, bottom: np.array) -> np.array: # 強烈光源
    top, bottom = top / 255.0, bottom / 255.0
    res = np.where(bottom <= 0.5,
                   1 - (1 - top) / (2 * bottom + 0.001),  # prevent ZeroDivisionError
                   top / (2 * (1 - bottom) + 0.001)) # prevent ZeroDivisionError
    res = np.clip(res, 0, 1)
    res = (255 * res).astype(np.uint8)
    return res

@apply_alpha
def linear_light(top: np.array, bottom: np.array) -> np.array: # 線性光源
    top, bottom = top / 255.0, bottom / 255.0
    res = np.clip(2 * top + bottom - 1.0, 0, 1)
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def lighten(top: np.array, bottom: np.array) -> np.array: # 變亮
    top, bottom = top / 255.0, bottom / 255.0
    res = np.maximum(top, bottom)
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def darken(top: np.array, bottom: np.array) -> np.array: # 變暗
    top, bottom = top / 255.0, bottom / 255.0
    res = np.minimum(top, bottom)
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def darker_color(top: np.array, bottom: np.array) -> np.array: # 顏色變暗
    top, bottom = top / 255.0, bottom / 255.0
    res = np.where(np.sum(top, axis=2, keepdims=True) < np.sum(bottom, axis=2, keepdims=True),
             top,
             bottom)
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def lighter_color(top: np.array, bottom: np.array) -> np.array: # 顏色變亮
    top, bottom = top / 255.0, bottom / 255.0
    res = np.where(np.sum(top, axis=2, keepdims=True) > np.sum(bottom, axis=2, keepdims=True),
             top,
             bottom)
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def pin_light(top: np.array, bottom: np.array) -> np.array: # 小光源
    top, bottom = top / 255.0, bottom / 255.0
    res = np.where(top <= 2 * bottom - 1,
             2 * bottom - 1,
             np.where(top >= 2 * bottom,
                  2 * bottom,
                  top))
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def hard_mix(top: np.array, bottom: np.array) -> np.array: # 實線疊印混合
    top, bottom = top / 255.0, bottom / 255.0
    res = np.where(top + bottom >= 1,
             1,
             0)
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def divide(top: np.array, bottom: np.array) -> np.array:
    top, bottom = top / 255.0, bottom / 255.0
    res = np.clip(bottom / (top + 0.001), 0, 1)  # prevent ZeroDivisionError
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def difference(top: np.array, bottom: np.array) -> np.array: # 差異化
    top, bottom = top / 255.0, bottom / 255.0
    res = np.abs(bottom - top)
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def exclusion(top: np.array, bottom: np.array) -> np.array: # 排除
    top, bottom = top / 255.0, bottom / 255.0
    res = (top + bottom) - 2 * top * bottom
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def subtract(top: np.array, bottom: np.array) -> np.array:
    top, bottom = top / 255.0, bottom / 255.0
    res = np.clip(bottom - top, 0, 1)
    res = (res * 255).astype(np.uint8)
    return res

@apply_alpha
def hue(top: np.array, bottom: np.array) -> np.array: # 色相
    top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    bottom_hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
    bottom_hsv[:,:,0] = top_hsv[:,:,0]
    res = cv2.cvtColor(bottom_hsv, cv2.COLOR_HSV2BGR)
    return res

@apply_alpha
def saturation(top: np.array, bottom: np.array) -> np.array: # 飽和度
    top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    bottom_hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
    bottom_hsv[:,:,1] = top_hsv[:,:,1]
    res = cv2.cvtColor(bottom_hsv, cv2.COLOR_HSV2BGR)
    return res

@apply_alpha
def color(top: np.array, bottom: np.array) -> np.array: # 顏色
    top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    bottom_hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
    bottom_hsv[:,:,0] = top_hsv[:,:,0]
    bottom_hsv[:,:,1] = top_hsv[:,:,1]
    res = cv2.cvtColor(bottom_hsv, cv2.COLOR_HSV2BGR)
    return res

@apply_alpha
def luminosity(top: np.array, bottom: np.array) -> np.array: # 明度
    top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    bottom_hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
    bottom_hsv[:,:,2] = top_hsv[:,:,2]
    res = cv2.cvtColor(bottom_hsv, cv2.COLOR_HSV2BGR)
    return res

blending_modes = {
    "normal": normal,
    "dissolve": dissolve,
    "multiply": multiply,
    "screen": screen,
    "overlay": overlay,
    "soft light": soft_light,
    "hard light": hard_light,
    "color dodge": color_dodge,
    "linear dodge": linear_dodge,
    "color burn": color_burn,
    "linear burn": linear_burn,
    "vivid light": vivid_light,
    "linear light": linear_light,
    "lighten": lighten,
    "darken": darken,
    "darker color": darker_color,
    "lighter color": lighter_color,
    "pin light": pin_light,
    "hard mix": hard_mix,
    "divide": divide,
    "difference": difference,
    "exclusion": exclusion,
    "subtract": subtract,
    "hue": hue,
    "saturation": saturation,
    "color": color,
    "luminosity": luminosity
}

def blurred(image: Image) -> Image:
    kernel_size = min(image.size) // 234
    kernel_size += 1 if kernel_size % 2 == 0 else 0
    # ref: https://blog.csdn.net/qq_16184125/article/details/107693222
    return image.filter(ImageFilter.GaussianBlur(kernel_size))


def paste_quadrilateral_ad(frame: Image, ad: Image, quadrilateral: np.array, mask: np.array) -> Image:

    ad_array = np.array(ad.convert("RGB"))
    frame_array = np.array(frame.convert("RGB"))
    blurred_frame_array = np.array(blurred(frame).convert("RGB"))

    # make quadrilateral mask
    quadrilateral_mask = np.zeros(frame_array.shape[:2], dtype=np.int32) # cv2.fillPoly can only accept int np.array
    cv2.fillPoly(quadrilateral_mask, [quadrilateral.astype(np.int32)], 255) # fill in polygon w/ 255 (True)
    quadrilateral_mask = quadrilateral_mask.astype(bool) # convert quadrilateral_mask to bool type
    quadrilateral_mask &= mask # take intersection w/ background mask

    h, w = ad_array.shape[:2]
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32").reshape(-1, 1, 2) # source points of ad
    dst_pts = quadrilateral.reshape(-1, 1, 2).astype(np.float32) # destination points of ad
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts) # get perspective transformation matrix of ad
    ad_array = cv2.warpPerspective(ad_array, matrix, (frame_array.shape[1], frame_array.shape[0]), None, cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT) # make ad fit the selected quadrilateral

    """light rendering way 1"""
    frame_array[quadrilateral_mask] = blending_modes["normal"](ad_array, frame_array)[quadrilateral_mask]
    frame_array[quadrilateral_mask] = blending_modes["overlay"](blurred_frame_array, frame_array, alpha=.9)[quadrilateral_mask]
    frame_array[quadrilateral_mask] = blending_modes["exclusion"](ad_array, frame_array, alpha=.15)[quadrilateral_mask]
    frame_array[quadrilateral_mask] = blending_modes["multiply"](blurred_frame_array, frame_array, alpha=.9)[quadrilateral_mask]
    frame_array[quadrilateral_mask] = blending_modes["screen"](blurred_frame_array, frame_array, alpha=.2)[quadrilateral_mask]
    frame_array[quadrilateral_mask] = blending_modes["saturation"](blurred_frame_array, frame_array, alpha=.15)[quadrilateral_mask]

    """light rendering way 2"""
    # ad_array1 = blending_modes["soft light"](ad_array, blurred_frame_array)
    # ad_array2 = blending_modes["multiply"](ad_array, blurred_frame_array)
    # frame_array[quadrilateral_mask] = blending_modes["normal"](ad_array2, frame_array, alpha=1)[quadrilateral_mask]
    # frame_array[quadrilateral_mask] = blending_modes["normal"](ad_array1, frame_array, alpha=.5)[quadrilateral_mask]

    """light rendering way 3"""
    # frame_array[quadrilateral_mask] = blending_modes["normal"](ad_array, frame_array)[quadrilateral_mask]
    # frame_array[quadrilateral_mask] = blending_modes["multiply"](blurred_frame_array, frame_array, alpha=.55)[quadrilateral_mask]
    # frame_array[quadrilateral_mask] = blending_modes["screen"](blurred_frame_array, frame_array, alpha=.55)[quadrilateral_mask]
    # frame_array[quadrilateral_mask] = blending_modes["color"](ad_array, frame_array, alpha=.9)[quadrilateral_mask]
    # frame_array[quadrilateral_mask] = blending_modes["color burn"](blurred_frame_array, frame_array, alpha=.4)[quadrilateral_mask]
    # frame_array[quadrilateral_mask] = blending_modes["saturation"](blurred_frame_array, frame_array, alpha=.75)[quadrilateral_mask]

    """light rendering way 4"""
    # frame_array[quadrilateral_mask] = blending_modes["normal"](ad_array, frame_array)[quadrilateral_mask]
    # frame_array[quadrilateral_mask] = blending_modes["exclusion"](ad_array, frame_array, alpha=.25)[quadrilateral_mask]
    # frame_array[quadrilateral_mask] = blending_modes["multiply"](blurred_frame_array, frame_array, alpha=.5)[quadrilateral_mask]
    # frame_array[quadrilateral_mask] = blending_modes["overlay"](blurred_frame_array, frame_array, alpha=.75)[quadrilateral_mask]

    return Image.fromarray(frame_array)

def show_quadrilateral(coords: np.array, ax, edgecolor='green'):
    assert len(coords) == 4 and coords.shape == (4, 2)
    poly = Polygon(coords, edgecolor=edgecolor, facecolor='none', lw=2)
    ax.add_patch(poly)

def get_largest_inner_box(mask:np.array, target_ratio=.00, tolerance=.01) -> np.array:
    """
    Get the largest inner box from the segmentation mask.

    Args:
    mask (np.array): The segmentation mask.
    target_ratio = (float): The target ratio of the largest inner box.
    tolerance (float): The tolerance of the largest inner box.

    Returns:
    np.array: The coordinates of the largest inner box.
    """
    rows, cols = mask.shape
    max_area = 0
    max_rect = (0, 0, 0, 0)  # (x1, y1, x2, y2)

    # ref: LeetCode 84. Largest Rectangle in Histogram
    heights = np.zeros(cols)
    for y2 in range(rows):
        if (rows - y2) * cols < max_area: # The rest area is smaller than the max area
            break

        heights = (heights + 1) * mask[y2, :] # compute the heights (histogram)
        stack = [] # reset stack at every row
        for x2 in range(cols + 1):
            if (rows - y2) * (cols + 1 - x2) < max_area: # The rest area is smaller than the max area
                break

            while stack and (x2 == cols or heights[stack[-1]] > heights[x2]):
                height = heights[stack.pop()]
                width = x2 if not stack else x2 - stack[-1] - 1
                curr_ratio = width / height if height > 0 else 0 # prevent ZeroDivisionError
                if target_ratio * (1 - tolerance) <= curr_ratio <= target_ratio * (1 + tolerance) or target_ratio == 0:
                    area = width * height
                    if area > max_area:
                        max_area = area
                        x1 = 0 if not stack else stack[-1] + 1
                        max_rect = (x1, y2 - height + 1, x2 - 1, y2)

            stack.append(x2)

     # Calculate the 20% smaller rectangle
    x1, y1, x2, y2 = max_rect
    width = x2 - x1 + 1
    height = y2 - y1 + 1

    # Reduce width and height by 20%
    new_width = int(width * 0.9)
    new_height = int(height * 0.9)

    # Calculate new coordinates ensuring they are within bounds
    new_x1 = x1 + (width - new_width) // 2
    new_y1 = y1 + (height - new_height) // 2
    new_x2 = new_x1 + new_width - 1
    new_y2 = new_y1 + new_height - 1

    # Ensure coordinates are within bounds
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(cols - 1, new_x2)
    new_y2 = min(rows - 1, new_y2)

    return np.array([new_x1, new_y1, new_x2, new_y2], dtype="uint32")

def show_mask(mask, ax):
    """
    Show the mask.

    Args:
    mask (np.array): The mask.
    ax (matplotlib.axes._subplots.AxesSubplot): The axis to show the mask.
    """
    color = np.array([30/255, 144/255, 255/255, 0.5])
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)