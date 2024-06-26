import os
import numpy as np
import cv2
from typing import List, Tuple
from numpy.typing import NDArray
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
import torch
from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines
from Process_package import get_mask, paste_quadrilateral_ad
from Process_package import get_largest_inner_box
from PIL import Image, ImageFilter


def line_in_rect(line, rect):
    """
    檢查線段是否完全或部分包含在矩形內

    參數:
    line: 每條預測出來的線
    rect: 我們的
    """
    x0, y0, x1, y1 = rect
    line_x1, line_y1, line_x2, line_y2 = line[0][0], line[0][1], line[1][0], line[1][1]

    # 檢查線段的兩個端點是否至少有一個在矩形內
    if (x0 <= line_x1 <= x1 and y0 <= line_y1 <= y1) or \
       (x0 <= line_x2 <= x1 and y0 <= line_y2 <= y1):
        return True

    # 檢查線段是否與矩形的任何一邊相交
    if line_intersects_side(line, (x0, y0, x1, y0)) or \
       line_intersects_side(line, (x1, y0, x1, y1)) or \
       line_intersects_side(line, (x1, y1, x0, y1)) or \
       line_intersects_side(line, (x0, y1, x0, y0)):
        return True

    return False

def line_intersects_side(line, side):
    """
    檢查線段是否與線段相交
    """
    x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
    x3, y3, x4, y4 = side

    # 計算兩條線段的斜率和截距
    try:
        m1 = (y2 - y1) / (x2 - x1)
    except ZeroDivisionError:
        m1 = np.inf
    b1 = y1 - m1 * x1

    try:
        m2 = (y4 - y3) / (x4 - x3)
    except ZeroDivisionError:
        m2 = np.inf
    b2 = y3 - m2 * x3

    # 如果斜率相同,則兩條線段平行,不相交
    if m1 == m2:
        return False

    # 計算兩條線段的交點
    try:
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    except (ZeroDivisionError, RuntimeWarning):
        # 如果發生除零錯誤或運行時警告,則視為不相交
        return False

    # 檢查交點是否在兩條線段上
    if min(x1, x2) <= x <= max(x1, x2) and min(x3, x4) <= x <= max(x3, x4) and \
       min(y1, y2) <= y <= max(y1, y2) and min(y3, y4) <= y <= max(y3, y4):
        return True

    return False

def outlier(data_to_pro): #去除斜率outlier
    data_np = np.array(data_to_pro)

    # 計算第一個和第三個四分位數
    q1 = np.percentile(data_np, 25)
    q3 = np.percentile(data_np, 75)

    # 計算四分位數間距
    iqr = q3 - q1

    # 定義上下限
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(f'上限為{upper_bound}, 下限為{lower_bound}')
    # 篩選異常值
    filtered_data = [x for x in data_np if x >= lower_bound and x <= upper_bound]

    return filtered_data

def show_box(box, ax, edgecolor='green'):
    """
    Show bounding box on the plot.

    Args:
    box: The bounding box coordinates.
    ax: The axis of the plot.
    edgecolor (str): The color of the bounding box. Defaults to 'green'.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2))


def find_line(predlines, innerbox):
    """ 
    找出包含在框框內的線段
    Args:
        predlines: 預測出來的線段
        innerbox: 框框的座標
    Returns:
        dict: 包含符合條件的線段和對應的斜率
    """
    lines_in_rect = []
    slopes = []
    for line in predlines:
        if line_in_rect(line, innerbox):
            lines_in_rect.append(line)
            x1, y1 = line[0]
            x2, y2 = line[1]
            try:
                slope = (y2 - y1) / (x2 - x1)
            except ZeroDivisionError:
                slope = np.inf
            slopes.append(slope)
    
    result = {'lines': lines_in_rect, 'slopes': slopes}
    return result

def classify_lines_with_slope(out: dict) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    lines = out['lines'][0]
    x1, y1 = lines[:, 0, 0], lines[:, 0, 1]
    x2, y2 = lines[:, 1, 0], lines[:, 1, 1]

    vertical_mask = x2 == x1
    slopes = np.where(vertical_mask, np.Infinity, (y2 - y1) / (x2 - x1))
    horizontal_mask = np.abs(slopes) <= 1

    vertical_lines = lines[vertical_mask | ~horizontal_mask]
    horizontal_lines = lines[horizontal_mask]

    return horizontal_lines, vertical_lines

def classify_lines_with_level(out: dict, tolerance=0.25) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    lines = out['lines'][0]
    line_level = out["line_level"].squeeze().cpu()

    lower_bound = 0.0
    upper_bound = np.pi  # max of line_level is π
    atol = (upper_bound - lower_bound) * tolerance
    mid_coords = (lines[:, 0, :] + lines[:, 1, :]) / 2
    mid_coords = mid_coords.astype(int)

    mid_lvls = line_level[mid_coords[:, 1], mid_coords[:, 0]]

    horizontal_mask = np.isclose(mid_lvls, lower_bound, atol=atol) | np.isclose(mid_lvls, upper_bound, atol=atol)
    horizontal_lines = lines[horizontal_mask]
    vertical_lines = lines[~horizontal_mask]

    return horizontal_lines, vertical_lines

def show_visualized_info(out: dict, info_keys=["df_norm", "df", "line_level"]):
  for info_key in info_keys:
    if info_key != 'lines':
      df = out[info_key]
      df = df.cpu()
      df_2d = df.squeeze()
      plt.figure(figsize=(10, 6))
      plt.imshow(df_2d, cmap='gray_r' if info_key == "df_norm" else 'gray')
      plt.colorbar()
      plt.axis('off')
      plt.show()

from functools import reduce
def Average(lst): 
    """ 
    use to calculate list average
    """
    return reduce(lambda a, b: a + b, lst) / len(lst) 

def find_line(predlines, innerbox):
    """ 
    找出包含在框框內的線段
    Args:
        predlines: 預測出來的線段
        innerbox: 框框的座標
    Returns:
        dict: 包含符合條件的線段和對應的斜率
    """
    lines_in_rect = []
    slopes = []
    for line in predlines:
        if line_in_rect(line, innerbox):
            lines_in_rect.append(line)
            x1, y1 = line[0]
            x2, y2 = line[1]
            try:
                slope = (y2 - y1) / (x2 - x1)
            except ZeroDivisionError:
                slope = float('inf')
            slopes.append(slope)
    
    result = {'lines': lines_in_rect, 'slopes': slopes}
    return result

def find_nearest_longest_horizontal_line(x, y, horiz_lines):
    """
    在 horizontal_lines 中找出離點 (x, y) 最近且最長的線段,並返回該線段的斜率和與該點的距離。
    Args:
        x (float): 目標點的 x 座標
        y (float): 目標點的 y 座標
        horizontal_lines (numpy.ndarray): 水平線段的座標陣列
    Returns:
        tuple: (slope, distance)
            slope (float): 最近最長線段的斜率
            distance (float): 最近最長線段與目標點的距離
    """
    distances = {}
    for line in horiz_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]

        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
        # 計算線段與目標點的距離
        distance = np.linalg.norm(np.cross(line[1] - line[0], np.array([x - line[0][0], y - line[0][1]])) / np.linalg.norm(line[1] - line[0]))
        line_length = np.linalg.norm(np.array(line[1]) - np.array(line[0]))
        distances[(tuple(line[0]), tuple(line[1])), line_length] = distance

    if not distances:
        return None, None

    # 根據距離和線段長度進行排序,取出最近且最長的線段
    sorted_distances = sorted(distances.items(), key=lambda x: (x[1], -x[0][1]))
    nearest_longest_line, line_length = sorted_distances[0][0]
    min_distance = sorted_distances[0][1]

    x1, y1 = nearest_longest_line[0]
    x2, y2 = nearest_longest_line[1]
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf

    return slope, min_distance

def find_nearest_horizontal_line(x, y, horiz_lines):
    min_distance = float('inf')
    nearest_line = None
    
    # 計算所有線段的平均長度
    line_lengths = []
    for line in horiz_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        line_lengths.append(line_length)
    avg_line_length = np.mean(line_lengths)

    for line in horiz_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if line_length >= avg_line_length:
            distance = np.linalg.norm(np.cross(line[1] - line[0], np.array([x - line[0][0], y - line[0][1]])) / np.linalg.norm(line[1] - line[0]))
            if distance < min_distance:
                min_distance = distance
                nearest_line = line
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf

    if nearest_line is None:
        return None, None
    
    return slope, min_distance

def compute_adjusted_region_points(x, y, d, m):
    """
    計算調整後的區域點座標(x1, y1)
    
    參數:
    x (float): 原始點的x座標
    y (float): 原始點的y座標
    d (float): 原始點到LETR線段的距離
    m (float): LETR線段的斜率
    
    返回:
    x1 (float): 調整後的x座標
    y1 (float): 調整後的y座標
    """
    r = math.sqrt(1 + m**2)  # 計算公式(1)
    x1 = x + d / r  # 計算調整後的x座標，公式(2)
    y1 = y + d * m / r  # 計算調整後的y座標，公式(2)
    return x1, y1


def find_min_distance_endpoints(box_x, box_y, merge_list, index):
    """
    在給定矩形區域內，找到與線段端點距離最短的線段。

    Args:
    box_x (float): 矩形區域的 X 軸中心座標。
    box_y (float): 矩形區域的 Y 軸中心座標。
    merge_list (list): 包含線段端點的列表，每個元素為一個包含兩個元組的列表，代表線段的兩個端點座標。
    index (int): 索引值，用於決定搜索上方或下方的線段。

    Returns:
    tuple: 包含以下元素的 tuple：
        - min_endpoints (list): 最短距離的線段端點座標。
        - min_distance (float): 最短距離值。
        - slope (float): 最短距離對應的線段的斜率。

    Example:
    >>> box_x = 0
    >>> box_y = 0
    >>> merge_list = [[(-1, -1), (1, 1)], [(2, 2), (3, 3)], [(-1, 1), (1, -1)]]
    >>> find_min_distance_endpoints(box_x, box_y, merge_list, 0)
    ([(2, 2), (3, 3)], 2.8284271247461903, 1.0)
    """
    # 初始化最小距離為正無窮大
    min_distance = float('inf')
    # 初始化最小距離對應的線段端點為空
    min_endpoints = None
    
    # 計算區域中心的 X 和 Y 座標
    region_center_x = box_x
    region_center_y = box_y
    
    # 遍歷線段列表中的每一條線段
    for line in merge_list:
        # 提取線段的第一個端點座標 (x1, y1) 和第二個端點座標 (x2, y2)
        x1, y1 = line[0]
        x2, y2 = line[1]
        
        # 計算區域中心與線段端點的距離
        dist1 = np.sqrt((x1 - region_center_x)**2 + (y1 - region_center_y)**2)
        dist2 = np.sqrt((x2 - region_center_x)**2 + (y2 - region_center_y)**2)
        
        # 計算線段的斜率
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
        
        # 根據索引決定搜索上方或下方的線段
        if index in [0, 1]:  # 如果索引為0或1，只找上方的線段
            # 檢查端點的 Y 座標是否小於區域中心的 Y 座標
            if y1 < region_center_y and y2 < region_center_y:
                # 檢查是否找到了更短的距離對
                if dist1 + dist2 < min_distance:
                    # 更新最小距離和最小距離對應的線段端點
                    min_distance = dist1 + dist2
                    min_endpoints = [(x1, y1), (x2, y2)]
        else:  # 其他索引值，只找下方的線段
            # 檢查端點的 Y 座標是否大於區域中心的 Y 座標
            if y1 > region_center_y and y2 > region_center_y:
                # 檢查是否找到了更短的距離對
                if dist1 + dist2 < min_distance:
                    # 更新最小距離和最小距離對應的線段端點
                    min_distance = dist1 + dist2
                    min_endpoints = [(x1, y1), (x2, y2)]
    
    # 返回最小距離對應的線段端點、最小距離值和對應線段的斜率
    return min_endpoints, min_distance, slope


def compute_adjusted_region_points(x, y, d, m):
    """
    計算調整後的區域點座標(x1, y1)
    
    參數:
    x (float): 原始點的x座標
    y (float): 原始點的y座標
    d (float): 原始點到LETR線段的距離
    m (float): LETR線段的斜率
    
    返回:
    x1 (float): 調整後的x座標
    y1 (float): 調整後的y座標
    """
    r = math.sqrt(1 + m**2)  # 計算公式(1)
    x1 = x + d / r  # 計算調整後的x座標，公式(2)
    y1 = y + d * m / r  # 計算調整後的y座標，公式(2)
    
    return x1, y1

def find_nearest_longest_horizontal_line(x, y, horiz_lines):
    """
    在 horizontal_lines 中找出離點 (x, y) 最近且最長的線段,並返回該線段的斜率和與該點的距離。
    Args:
        x (float): 目標點的 x 座標
        y (float): 目標點的 y 座標
        horizontal_lines (numpy.ndarray): 水平線段的座標陣列
    Returns:
        tuple: (slope, distance)
            slope (float): 最近最長線段的斜率
            distance (float): 最近最長線段與目標點的距離
    """
    distances = {}
    for line in horiz_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]

        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
        # 計算線段與目標點的距離
        distance = np.linalg.norm(np.cross(line[1] - line[0], np.array([x - line[0][0], y - line[0][1]])) / np.linalg.norm(line[1] - line[0]))
        line_length = np.linalg.norm(np.array(line[1]) - np.array(line[0]))
        distances[(tuple(line[0]), tuple(line[1])), line_length] = distance

    if not distances:
        return None, None

    # 根據距離和線段長度進行排序,取出最近且最長的線段
    sorted_distances = sorted(distances.items(), key=lambda x: (x[1], -x[0][1]))
    nearest_longest_line, line_length = sorted_distances[0][0]
    min_distance = sorted_distances[0][1]

    x1, y1 = nearest_longest_line[0]
    x2, y2 = nearest_longest_line[1]
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf

    return slope, min_distance
