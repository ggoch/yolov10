import cv2
import numpy as np
import math

# 忽略檔名中的標點符號
def clean_label(label):
    return ''.join(char for char in label if char.isalnum())

def nms(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if isinstance(indices, (list, tuple)):  # 檢查 indices 是否為列表或元組
        return [i[0] for i in indices]
    return indices.flatten().tolist()  # 否則，將其展平為列表

def remove_duplicate_boxes(detected_boxes):
    # 使用集合来存储唯一的坐标元组
    seen_coordinates = set()
    unique_boxes = []

    for box in detected_boxes:
        # 将坐标数组转换为不可变的元组
        coordinates_tuple = tuple([box[0], box[1], box[2], box[3]])
        
        # 检查元组是否已经在集合中
        if coordinates_tuple not in seen_coordinates:
            seen_coordinates.add(coordinates_tuple)
            unique_boxes.append(box)
    
    return unique_boxes

# 計算IOU（交并比）
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def calculate_distance(box1, box2):
    # 计算两个盒子中心点之间的欧几里得距离
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]
    center1 = (x1 + w1 / 2, y1 + h1 / 2)
    center2 = (x2 + w2 / 2, y2 + h2 / 2)
    return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

def remove_close_boxes(detected_boxes, threshold):
    filtered_boxes = []
    
    for box in detected_boxes:
        too_close = False
        for filtered_box in filtered_boxes:
            if calculate_distance(box, filtered_box) < threshold:
                too_close = True
                break
        if not too_close:
            filtered_boxes.append(box)
    
    return filtered_boxes

def splice_by_label(lst, label_to_remove):
    removed_item = None
    for i, item in enumerate(lst):
        if item[4] == label_to_remove:
            removed_item = lst.pop(i)
            break
    return removed_item

def process_result_result_car_no(results,names):
    """
    分析車牌檢測結果，去除重複文字，並返回檢測到的車牌號碼
    results: 檢測結果
    names: 模型類別名稱
    """
    detected_boxes = []
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        class_id = result['class_id']
        detected_char = result['class_name']
        confidence = result['confidence']
        # if detected_char != 'license plate':
        detected_boxes.append((x1, y1, x2, y2, detected_char, confidence))

    detected_boxes = remove_duplicate_boxes(detected_boxes)

    filtered_boxes = remove_close_boxes(detected_boxes, 10)

    license_plate = splice_by_label(filtered_boxes, 'license plate')


    # 根據x1座標排序檢測到的字符
    filtered_boxes.sort(key=lambda x: x[0])

    # 拼接檢測到的字符
    detected_label = ''.join([box[4] for box in filtered_boxes])

    return detected_label,filtered_boxes,license_plate