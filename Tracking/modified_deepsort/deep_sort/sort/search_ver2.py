import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# 这两个函数是为了删除部分被包含的方框
def is_inside(box1, box2):
    """检查box1是否被box2完全包围"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    return x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2)


def remove_inside_boxes(bounding_boxes):
    """删除所有被其他框完全包围的框"""
    to_remove = []

    for i in range(len(bounding_boxes)):
        for j in range(len(bounding_boxes)):
            if i != j and is_inside(bounding_boxes[i], bounding_boxes[j]):
                to_remove.append(bounding_boxes[i])
                break  # No need to check against other boxes as it is already marked for removal

    # Use set difference to get boxes that are not to be removed
    unique_boxes = set(bounding_boxes)
    retained_boxes = list(unique_boxes - set(to_remove))

    return retained_boxes
# 计算方框中心的欧氏距离
def euclidean_distance(box_tlwh, box_xywh):
    # 将box_tlwh从左上角格式转换到中心点格式
    center_tlwh = np.array([box_tlwh[0] + box_tlwh[2] / 2.0,
                            box_tlwh[1] + box_tlwh[3] / 2.0])

    # 计算两个中心点之间的欧氏距离
    distance = np.linalg.norm(center_tlwh - box_xywh[:2])

    return distance

# 抑制之前就被检测出的方框
def iou_suppression(detections, boxes, threshold):
    # 转换 detections 到 numpy 数组
    detection_boxes = np.array([detections[idx].tlwh for idx in range(len(detections))])
    detection_boxes_expanded = np.expand_dims(detection_boxes, axis=0)

    # 转换 boxes 到 numpy 数组并扩展维度
    boxes = np.asarray(boxes)
    boxes_expanded = np.expand_dims(boxes, axis=1)

    # 扩展矩阵并进行tile操作
    detection_boxes_tiled = np.tile(detection_boxes_expanded, (boxes_expanded.shape[0], 1, 1))
    boxes_tiled = np.tile(boxes_expanded, (1, detection_boxes_expanded.shape[1], 1))

    # 计算交集的位置
    intersection_tl = np.maximum(detection_boxes_tiled[:, :, :2], boxes_tiled[:, :, :2])
    intersection_br = np.minimum(detection_boxes_tiled[:, :, :2] + detection_boxes_tiled[:, :, 2:],
                                 boxes_tiled[:, :, :2] + boxes_tiled[:, :, 2:])
    intersection_wh = np.maximum(intersection_br - intersection_tl, 0)

    # 计算交集的面积
    intersection_area = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]
    detection_area = detection_boxes_tiled[:, :, 2] * detection_boxes_tiled[:, :, 3]
    boxes_area = boxes_tiled[:, :, 2] * boxes_tiled[:, :, 3]

    # 计算 IoU 值
    iou_matrix = intersection_area / (detection_area + boxes_area - intersection_area)

    # 对于每一个box，找到与之相交的detection的最大IoU
    max_iou_for_each_box = np.max(iou_matrix, axis=1)

    # 找到超过阈值的boxes
    suppress_mask = max_iou_for_each_box > threshold

    # 保留那些未被抑制的boxes
    retained_boxes = boxes[np.logical_not(suppress_mask)]

    return retained_boxes




# 填充孔洞
def imfill(input_img):
    '''
    Implement the 'imfill' operation from Matlab in Python using OpenCV.

    Parameters:
    - input_img: Binary image that needs holes to be filled.

    Returns:
    - im_out: Binary image with holes filled.
    '''
    # Add a border around the input image
    padded_img = cv2.copyMakeBorder(input_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Copy the padded image to ensure not to change the original image
    im_floodfill = padded_img.copy()

    # Get image dimensions
    h, w = padded_img.shape[:2]

    # Make a mask with dimensions two pixels larger in each dimension
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Apply flood fill from the point (0,0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert the flood-filled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine padded image and inverted flood-filled image
    im_out_padded = padded_img | im_floodfill_inv

    # Remove the added border
    im_out = im_out_padded[1:-1, 1:-1]

    return im_out
#删除过小的轮廓
def filter_contours(contour_img, area_threshold=None, length_threshold=None, aspect_ratio_threshold=2):
    _, binary_img = cv2.threshold(contour_img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(contour_img, dtype=np.uint8)
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)

        # 计算边界框的坐标
        x, y, w, h = cv2.boundingRect(contour)
        # 计算长宽比
        aspect_ratio = float(h) / w if w > 0 else 0  # 防止除以0的情况

        # 检查长宽比是否满足条件
        if aspect_ratio_threshold:
            if aspect_ratio > aspect_ratio_threshold or 1.0 / aspect_ratio > aspect_ratio_threshold:
                continue

        # 判断轮廓是否满足面积和长度的条件
        if (area_threshold is None or area >= area_threshold) and \
           (length_threshold is None or length >= length_threshold):
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    return mask, filtered_contours

def find_box(img):
    # 使用Sobel算子进行边缘检测
    img = img[:,:,0]
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)

    # 将sobel_combined转换为8-bit
    sobel_8bit = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    _, BWs = cv2.threshold(sobel_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 创建结构元素
    se90 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    se0 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

    # 使用结构元素进行膨胀
    BWsdil = cv2.dilate(BWs, se90)
    BWsdil = cv2.dilate(BWsdil, se0)

    # 使用imfill函数
    filled_img = imfill(BWsdil)

    threshold_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold_img = cv2.bitwise_not(threshold_img)
    mask, filtered_contours = filter_contours(threshold_img, area_threshold=20, length_threshold=30)

    # 使用距离变换确定前景
    dist_transform = cv2.distanceTransform(filled_img, cv2.DIST_L2, 5)
    # Normalize the distance image for range = {0.0, 1.0}
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # 得到mask和filled_img的交集
    intersection_img = cv2.bitwise_and(mask, filled_img)
    intersection_img, filtered_contours = filter_contours(intersection_img, area_threshold=60, length_threshold=30)

    # 准备分水岭算法
    unknown = cv2.subtract(filled_img, intersection_img)

    # 标记前景为1
    ret, markers = cv2.connectedComponents(intersection_img.astype(np.uint8), connectivity=4)

    # 把背景标为0，不确定区域标为-1
    markers = markers + 1
    markers[unknown == 255] = 0

    # 使用分水岭算法
    markers = cv2.watershed(cv2.merge([img, img, img]), markers)

    # 基于标记得到前景、背景和不确定的区域
    foreground = np.zeros_like(img, dtype=np.uint8)
    background = np.zeros_like(img, dtype=np.uint8)
    uncertain = np.zeros_like(img, dtype=np.uint8)

    foreground[markers == -1] = 255
    background[markers == 1] = 255
    uncertain[(markers > 1) & (markers != -1)] = 255

    # 创建分割结果的显示
    segmented = np.zeros_like(cv2.merge([img, img, img]))

    # 使用蓝色标记背景，绿色标记前景，其他为红色
    segmented[markers == -1] = [0, 0, 255]  # red for uncertain
    segmented[markers == 1] = [255, 0, 0]  # blue for background
    segmented[(markers > 1) & (markers != -1)] = [0, 255, 0]  # green for foreground

    # 找到前景对象的边界
    foreground_mask = np.isin(markers, [2, 3, 4,
                                        5])  # you might need to adjust these numbers depending on how many objects you've segmented

    # 使用connectedComponents来找四联通的组件
    num_labels, labels = cv2.connectedComponents(foreground_mask.astype(np.uint8), connectivity=4)
    bounding_boxes = []
    for i in range(1, num_labels):  # 开始从1遍历，因为0是背景标签
        r, c = np.where(labels == i)
        if len(r) == 0 or len(c) == 0:  # 如果没有匹配到任何标签，跳过
            continue
        x, y, w, h = np.min(c), np.min(r), np.max(c) - np.min(c), np.max(r) - np.min(r)
        bounding_boxes.append((x, y, w, h))
    final_boxes = remove_inside_boxes(bounding_boxes)
    final_boxes = [box for box in final_boxes if 250 <= box[2] * box[3] <= 1600]
    return final_boxes

call_count = 0
def get_box(track, detections, img):
    global call_count
    call_count += 1
    x, y, aspect_ratio, h = track.mean[:4]
    h = 100
    w = 100
    left = int(x - w / 2)
    top = int(y - h / 2)
    right = int(x + w / 2)
    bottom = int(y + h / 2)
    # 判断是否需要调整裁剪区域的大小
    if left < 0:
        right += abs(left)
        left = 0
    if right > img.shape[1]:
        left -= (right - img.shape[1])
        right = img.shape[1]
    if top < 0:
        bottom += abs(top)
        top = 0
    if bottom > img.shape[0]:
        top -= (bottom - img.shape[0])
        bottom = img.shape[0]

    # 检查裁剪区域是否为空，如果为空，直接返回空数组
    if top == bottom or left == right:
        return []

    # 裁剪图像
    cropped_img = img[top:bottom, left:right]
    plt.imsave(f"D:\\cropped_img\\{call_count}.jpg", cropped_img, cmap='gray')
    # 在裁剪后的图像中查询分割出的图像，注意这里得到的Box的格式是左上角坐标+wh值
    boxes = find_box(cropped_img)

    if len(boxes) > 0:
        # 转换boxes到原图中的坐标
        boxes_in_original_image = [(box[0] + left, box[1] + top, box[2], box[3]) for box in boxes]
        # 先进行一次抑制来删除掉那些已经被检测出来的框
        boxes_in_original_image = iou_suppression(detections, boxes_in_original_image, 0.3)

        distance_values = []
        for box in boxes_in_original_image:
            distance = euclidean_distance(box, track.mean[:4])
            distance_values.append(distance)
        # print('这次的距离是',distance_values)

        if len(distance_values) > 0:
            min_distance_idx = np.argmin(distance_values)
            closest_box = boxes_in_original_image[min_distance_idx]

            # print('最近的Box',closest_box)
        else:
            closest_box = []

        return closest_box

    return []