%%time
import numpy as np
from adjustment_package import classify_lines_with_slope, find_nearest_longest_horizontal_line, \
    compute_adjusted_region_points
from tqdm import tqdm

ad_name = 'pic/recording_room.jpg'
ad_img = Image.open(ad_name)
w, h = ad_img.size
target_ratio = w / h


###################################the new way code
def Box_to_Segmentation(points):
    mask_predictor = SamPredictor(SAM)
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)

    width = max_x - min_x
    height = max_y - min_y

    # 生成 default_box
    default_box = {'x': min_x, 'y': min_y, 'width': width, 'height': height, 'label': ''}
    box = default_box
    box = np.array([
        box['x'],
        box['y'],
        box['x'] + box['width'],
        box['y'] + box['height']
    ])

    # Predictor masks
    image_bgr = cv2.imread('recording_room.jpg')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask_predictor.set_image(image_rgb)

    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=True
    )

    return masks


##Define a function to rasterize lines
def rasterize_line(img, line):
    pt1, pt2 = (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1]))
    cv2.line(img, pt1, pt2, color=1, thickness=1)


def find_intersecting_lines(masks_find_interline, pred_lines_find_interline):
    ##Create a binary mask image
    height, width = 1024, 1024  # Adjust based on your actual image dimensions
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    mask_coords = np.argwhere(masks_find_interline == 1)
    for coord in mask_coords:
        binary_mask[coord[1], coord[2]] = 1  # Set the pixels of the mask to 1

    line_segments = pred_lines_find_interline
    intersecting_lines = []
    for line in line_segments:
        line_img = np.zeros_like(binary_mask)
        rasterize_line(line_img, line)
        if np.any(np.logical_and(binary_mask, line_img)):
            intersecting_lines.append(line)
    return intersecting_lines


def find_2longest_line_length(linelist_for_compute):
    line_length = np.linalg.norm(line[1] - line[0])
    line_lengths = [(line, line_length(line)) for line in linelist_for_compute]
    # Sort the lines by length in descending order
    sorted_lines = sorted(line_lengths, key=lambda x: x[1], reverse=True)

    # Get the two longest lines
    longest_lines = sorted_lines[:2]

    # Extract the line coordinates from the result
    longest_lines_forreturn = [line[0] for line in longest_lines]
    return longest_lines_forreturn


def compute_vanishing_point(line1, line2):
    # 轉換線段到直線方程式的參數形式 Ax + By + C = 0
    def line_params(p1, p2):
        A = p1[1] - p2[1]
        B = p2[0] - p1[0]
        C = p1[0] * p2[1] - p2[0] * p1[1]
        return A, B, -C

    # 提取兩條線的端點
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # 計算直線方程式的參數
    A1, B1, C1 = line_params((x1, y1), (x2, y2))
    A2, B2, C2 = line_params((x3, y3), (x4, y4))

    # 建立係數矩陣和常數向量
    A = np.array([[A1, B1], [A2, B2]])
    C = np.array([C1, C2])

    # 解方程組 Ax = -C
    try:
        point = np.linalg.solve(A, -C)
    except np.linalg.LinAlgError:
        # 如果矩陣不可逆，說明兩條線是平行的，沒有交點
        return None

    return point


#########################################################################################
def sharpen_image(image):
    """Apply a sharpening filter to the image."""
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    return sharpened


def get_mask(img):
    img_arr = np.array(img)[:, :, 3]  # get alpha only
    img_arr = (img_arr > .5).astype(bool)
    return img_arr


def process_video(video_path, output_path):
    # 打開影片
    cap = cv2.VideoCapture(video_path)
    # 獲取影片屬性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 創建輸出影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 計算第一楨的loc
    ret, frame_np = cap.read()
    frame = Image.fromarray(frame_np)
    loc = get_loc_for_frame(frame, ad_img, target_ratio)
    i = 0
    pbar = tqdm(total=total_frames, unit='frames')

    while True:
        # 讀取一個影格
        ret, frame_np = cap.read()

        if not ret:
            # 到達影片結尾,退出循環
            break

        # 處理這一個影格
        sharpened_frame_np = sharpen_image(frame_np)
        frame = Image.fromarray(sharpened_frame_np)
        processed_frame = process_frame(frame, ad_img, loc, mask=None)
        if i == 1:
            plt.imshow(processed_frame)
            plt.show()
        # 將處理後的影格寫入輸出影片
        out.write(processed_frame)
        i += 1
        pbar.update(1)

    # 釋放資源
    cap.release()
    out.release()


def get_loc_for_frame(frame, ad_img, target_ratio):
    # 生成遮罩
    depthmap = get_depthmap_new(frame, model_holder)
    # try to binarize the output depthmap for finding errors
    binarized_img = binarized(depthmap, target_ratio=TARGET_RATIO)
    # bounding box
    bounding_box = find_min_bounding_box_from_img(binarized_img)
    masks = get_result_from_SAM(frame, bounding_box)
    mask = masks[0] if USE_DEPTHMAP else masks[0]["segmentation"]
    color = np.array([0, 0, 0, 1.0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask = get_mask(mask_image)

    # 檢測線條&找出線條之斜率
    points = [(2, 37), (430, 295), (427, 104), (0, 323)]  # 由使用者UI選擇
    masks = Box_to_Segmentation(points)
    ##deeplsd 檢測線條
    gray_img = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
    inputs = {'image': torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.}
    with torch.no_grad():
        out = net(inputs)
        pred_lines = out['lines'][0]
    ##找出跟遮罩相鄰的所有線
    intersecting_lines = find_intersecting_lines(masks, pred_lines)

    ##將與遮罩相鄰的所有線分別出垂直與平行
    horizontal_list, vertical_list = classify_lines_with_slope(intersecting_lines)

    ##找出平行線中最長的兩條線!!!!!!!!!!!!!!!!!!!!!!!(邏輯可能錯誤使得此方法無法應用至其他場域)(待修改)
    long2_horizontal_list = find_2longest_line_length(horizontal_list)
    ##計算消失點

    # 打開原始圖像
    original_img = Image.fromarray(cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB))

    # 找到最大內部框
    largest_inner_box = get_largest_inner_box(mask, target_ratio, box_size_ratio=0.6)
    inner_box = [[largest_inner_box[0], largest_inner_box[1]], [largest_inner_box[2], largest_inner_box[1]],
                 [largest_inner_box[0], largest_inner_box[3]], [largest_inner_box[2], largest_inner_box[3]]]
    rect = ((largest_inner_box[0], largest_inner_box[1]), (largest_inner_box[2], largest_inner_box[3]))
    edges = [(tuple(line[:2]), tuple(line[2:])) for line in pred_lines]

    up_line, down_line = find_transformation_lines(rect, horizontal_list)
    print(up_line)

    ## adj
    top_edge = [inner_box[0], inner_box[1]]
    bottom_edge = [inner_box[2], inner_box[3]]
    # 調整上邊
    new_top_edge = mcompute_adjusted_region_points(top_edge, up_line)

    # 調整下邊
    new_bottom_edge = mcompute_adjusted_region_points(bottom_edge, down_line)
    inner_box[0], inner_box[1] = new_top_edge
    inner_box[2], inner_box[3] = new_bottom_edge
    loc = np.array([inner_box[0], inner_box[1], inner_box[3], inner_box[2]])
    print(loc)
    return loc


def process_frame(frame, ad_img, loc, mask=None):
    if mask is None:
        mask = np.ones_like(np.array(frame)[:, :, 0], dtype=bool)
    # 將廣告貼到原始圖像上
    original_img = Image.fromarray(cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB))
    combined_image = paste_quadrilateral_ad(original_img, ad_img, loc, mask)
    combined_image_np = np.array(combined_image)[:, :, ::-1]
    return combined_image_np


# 使用示例
process_video('video/p_fo_video.mp4', 'outputvideo/output_p_fo_opdark_video.mp4')