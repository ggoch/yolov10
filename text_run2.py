import os
import json
import cv2
from ultralytics import YOLO

# 加載YOLO模型
model = YOLO('runs/detect/train39/weights/best.pt')

# 圖像資料夾路徑
image_folder = './car_license_fix'

# 存儲檢測結果和準確率的字典
results_summary = {
    'total_images': 0,
    'total_detections': 0,
    'correct_detections': 0
}

# 忽略檔名中的標點符號
def clean_label(label):
    return ''.join(char for char in label if char.isalnum())

def nms(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if isinstance(indices, (list, tuple)):  # 檢查 indices 是否為列表或元組
        return [i[0] for i in indices]
    return indices.flatten().tolist()  # 否則，將其展平為列表

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

# 遍歷圖像資料夾
for image_filename in os.listdir(image_folder):
    if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
        image_path = os.path.join(image_folder, image_filename)

        # 加載圖像
        image = cv2.imread(image_path)

        # 使用 YOLO 模型進行字符檢測
        results = model(image)

        # 從檔名中獲取正確答案並清理標點符號
        correct_label = clean_label(os.path.splitext(image_filename)[0].split('_')[0])

        detected_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                class_id = box.cls[0].int().item()
                confidence = box.conf[0].item()
                detected_char = model.names[class_id]
                if detected_char != 'license plate':  # 忽略 'license plate'
                    detected_boxes.append((x1, y1, x2, y2, detected_char, confidence))

        # 使用NMS去掉重疊度高的字符，保留信心度高的
        if detected_boxes:
            boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in detected_boxes]
            scores = [confidence for _, _, _, _, _, confidence in detected_boxes]
            indices = nms(boxes, scores, iou_threshold=0.7)
            detected_boxes = [detected_boxes[i] for i in indices]

        # 根據x1座標排序檢測到的字符
        detected_boxes.sort(key=lambda x: x[0])

        # 拼接檢測到的字符
        detected_label = ''.join([box[4] for box in detected_boxes])

        print(f"Filename: {image_filename}, Correct label: {correct_label}, Detected label: {detected_label}")

        results_summary['total_images'] += 1
        results_summary['total_detections'] += 1
        if detected_label == correct_label:
            results_summary['correct_detections'] += 1

# 計算準確率
accuracy = results_summary['correct_detections'] / results_summary['total_images'] if results_summary['total_images'] > 0 else 0

# 輸出結果
results_summary['accuracy'] = accuracy

# 將結果保存為 JSON 文件
summary_json_path = './results_summary.json'
with open(summary_json_path, 'w') as summary_file:
    json.dump(results_summary, summary_file, indent=4)

print(f"Accuracy: {accuracy * 100:.2f}%")