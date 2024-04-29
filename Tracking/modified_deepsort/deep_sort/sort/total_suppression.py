import numpy as np


def calculate_iou_matrix(detections, unmatched_detections_1, threshold):
    if len(unmatched_detections_1) == 0 or len(unmatched_detections_1) == len(detections):
        return unmatched_detections_1

    # 获取除了未匹配的索引外的其他索引，作为已匹配的索引
    match_indices = np.setdiff1d(np.arange(len(detections)), unmatched_detections_1)

    # 获取未匹配的框的位置和尺寸 (x, y, w, h)
    unmatched_bboxes = np.array([detections[idx].tlwh for idx in unmatched_detections_1])

    # 获取已匹配的框的位置和尺寸 (x, y, w, h)
    match_bboxes = np.array([detections[idx].tlwh for idx in match_indices])

    # 扩充未匹配的框矩阵和已匹配的框矩阵
    unmatched_bboxes_expanded = np.expand_dims(unmatched_bboxes, axis=1)
    match_bboxes_expanded = np.expand_dims(match_bboxes, axis=0)

    # 将两个扩充后的矩阵拼接在一起
    unmatched_bboxes_expanded = np.tile(unmatched_bboxes_expanded, (1, match_bboxes_expanded.shape[1], 1))
    match_bboxes_expanded = np.tile(match_bboxes_expanded, (unmatched_bboxes_expanded.shape[0], 1, 1))

    # 计算交集的位置和尺寸
    intersection_tl = np.maximum(unmatched_bboxes_expanded[:, :, :2], match_bboxes_expanded[:, :, :2])
    intersection_br = np.minimum(unmatched_bboxes_expanded[:, :, :2] + unmatched_bboxes_expanded[:, :, 2:],
                                 match_bboxes_expanded[:, :, :2] + match_bboxes_expanded[:, :, 2:])
    intersection_wh = np.maximum(intersection_br - intersection_tl, 0)

    # 计算交集和并集的面积
    intersection_area = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]
    unmatched_area = unmatched_bboxes_expanded[:, :, 2] * unmatched_bboxes_expanded[:, :, 3]
    match_area = match_bboxes_expanded[:, :, 2] * match_bboxes_expanded[:, :, 3]

    # 计算 IoU 值
    iou_matrix = intersection_area / (unmatched_area + match_area - intersection_area)

    # 对每一行的 IoU 值求和，并与阈值进行比较
    iou_sum = np.sum(iou_matrix, axis=1)
    exceeded_threshold = iou_sum > threshold

    # 从 unmatched_detections_1 中删除超过阈值的行对应的索引
    filtered_unmatched_detections_1 = np.delete(unmatched_detections_1, np.where(exceeded_threshold))

    return filtered_unmatched_detections_1
