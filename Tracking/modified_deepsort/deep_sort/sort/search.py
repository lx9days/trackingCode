import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def calculate_iou(box1, box2):
    # 提取方框1的坐标和尺寸
    x1, y1, aspect_ratio1, h1 = box1
    w1 = aspect_ratio1 * h1

    # 方框1的坐标为中心点，转换为左上角的坐标
    x1 = x1 - w1 / 2
    y1 = y1 - h1 / 2

    # 提取方框2的坐标和尺寸
    x2, y2, w2, h2 = box2

    # 计算交集的坐标
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # 如果两个方框没有交集，则IoU为0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集的面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集的面积
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    # 计算IoU值
    iou = intersection_area / union_area

    return iou



def filter_contours(contour_img, area_threshold=None, length_threshold=None):
    _, binary_img = cv2.threshold(contour_img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(contour_img, dtype=np.uint8)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)
        if (area_threshold is None or area >= area_threshold) and (length_threshold is None or length >= length_threshold):
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    return mask, filtered_contours

def find_box(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    threshold_img = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold_img = cv2.bitwise_not(threshold_img)
    mask, filtered_contours = filter_contours(threshold_img, area_threshold=20, length_threshold=30)
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((4, 4), np.uint8)
    dilated = cv2.dilate(mask, kernel1, iterations=3)
    filled = cv2.erode(dilated, kernel2, iterations=2)
    _, floodfilled, _, _ = cv2.floodFill(filled.copy(), None, seedPoint=(0, 0), newVal=255)
    floodfilled_inv = cv2.bitwise_not(floodfilled)
    out = filled | floodfilled_inv
    out = cv2.dilate(out, kernel1, iterations=1)
    sure_bg = cv2.bitwise_not(out)
    dist_transform = cv2.distanceTransform(out, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(out, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1

    # Set the boundary region to 0
    markers[unknown == 255] = 0

    # Perform the watershed
    markers = cv2.watershed(image, markers)

    n, m = markers.shape

    # Set the boundary region back to 1
    markers[0, :] = 1  # Top boundary
    markers[n - 1, :] = 1  # Bottom boundary
    markers[:, 0] = 1  # Left boundary
    markers[:, m - 1] = 1  # Right boundary

    image[markers == -1] = [0, 0, 255]

    bounding_boxes = []

    for i in np.unique(markers):
        if i <= 1:
            continue
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == i] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes
call_count = 0
def get_box(track, img, size, threshold):
    global call_count
    call_count += 1
    x, y, aspect_ratio, h = track.lastmean[:4]
    h = 80
    w = 80
    left = int(x - w/2)
    top = int(y - h/2)
    right = int(x + w/2)
    bottom = int(y + h/2)
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
    # 在裁剪后的图像中运行edge函数
    boxes = find_box(cropped_img)

    if len(boxes) > 0:
        # 转换boxes到原图中的坐标
        boxes_in_original_image = [(box[0]+left, box[1]+top, box[2], box[3]) for box in boxes]

        # 计算与原图中每个方框的IoU值
        iou_values = []
        for box in boxes_in_original_image:
            iou = calculate_iou(track.mean[:4], box)
            iou_values.append(iou)
        print(iou_values)

        # 找到大于阈值的最大IoU值的方框索引
        valid_iou_idx = np.where(np.array(iou_values) >  threshold)[0]
        if len(valid_iou_idx) > 0:
            max_iou_idx = valid_iou_idx[np.argmax(np.array(iou_values)[valid_iou_idx])]
            closest_box = boxes_in_original_image[max_iou_idx]
            #
            # # 转换closest_box坐标到裁剪图像
            # closest_box_cropped = (closest_box[0] - left, closest_box[1] - top, closest_box[2], closest_box[3])
            #
            # # 使用matplotlib绘制裁剪图像和方框
            # plt.imshow(cropped_img, cmap='gray')
            # rect = plt.Rectangle((closest_box_cropped[0], closest_box_cropped[1]), closest_box_cropped[2],
            #                      closest_box_cropped[3], linewidth=1, edgecolor='r', facecolor='none')
            # plt.gca().add_patch(rect)
            # plt.axis('off')
            # plt.show(block=False)
            # plt.pause(1)  # 展示1秒
            # plt.close()


            return closest_box

    return []

