from ultralytics import YOLOv10
import cv2
import json
import os

# 加載第一次檢測的YOLO模型
model1 = YOLOv10('runs/detect/train62/weights/best.pt')
# 加載第二次檢測的YOLO模型
model2 = YOLOv10('runs/detect/train74/weights/best.pt')

# 加載圖像
image_path = r'.\picture\FE62F4BC-4BB4-02AF-3641-3A0BE8CEB0CB_SwipeCard.jpg'
image = cv2.imread(image_path)

# 使用第一次檢測模型進行檢測
results1 = model1.predict(image, imgsz=640)

# 準備存儲檢測結果的字典
detection_results = {
    'image_path': image_path,
    'detections': []
}

# 繪製第一次檢測結果並存儲到字典中
license_plates = []
for result in results1:
    boxes = result.boxes.xyxy
    class_ids = result.boxes.cls
    scores = result.boxes.conf
    for box, class_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = box.int().tolist()
        class_id = class_id.int().item()
        score = score.item()
        detection_results['detections'].append({
            'bbox': [x1, y1, x2, y2],
            'class_id': class_id,
            'class_name': model1.names[class_id],
            'confidence': score
        })
        # 如果檢測到的類別是車牌，則記錄車牌區域
        if model1.names[class_id] == 'license plate':
            license_plates.append((x1, y1, x2, y2))
        # 在圖片上繪製第一次檢測的框和類別名稱
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{model1.names[class_id]}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 創建存儲車牌圖像的目錄
os.makedirs('./picture/license_plates', exist_ok=True)

# 針對每個車牌區域使用第二次檢測模型進行檢測並單獨存儲
for idx, (x1, y1, x2, y2) in enumerate(license_plates):
    license_plate_image = image[y1:y2, x1:x2]
    resized_license_plate_image = cv2.resize(license_plate_image, (160, 128))
    license_plate_image_path = f'./picture/license_plates/license_plate_{idx}.jpg'
    cv2.imwrite(license_plate_image_path, resized_license_plate_image)

    plate_results = model2.predict(resized_license_plate_image, imgsz=160)
    for plate_result in plate_results:
        plate_boxes = plate_result.boxes.xyxy
        plate_class_ids = plate_result.boxes.cls
        plate_scores = plate_result.boxes.conf
        for plate_box, plate_class_id, plate_score in zip(plate_boxes, plate_class_ids, plate_scores):
            px1, py1, px2, py2 = plate_box.int().tolist()
            plate_class_id = plate_class_id.int().item()
            plate_score = plate_score.item()
            detection_results['detections'].append({
                'bbox': [x1 + int(px1 * (x2 - x1) / 160), y1 + int(py1 * (y2 - y1) / 128), x1 + int(px2 * (x2 - x1) / 160), y1 + int(py2 * (y2 - y1) / 128)],
                'class_id': plate_class_id,
                'class_name': model2.names[plate_class_id],
                'confidence': plate_score
            })
            # 在原圖上繪製第二次檢測的框和類別名稱
            cv2.rectangle(image, (x1 + int(px1 * (x2 - x1) / 160), y1 + int(py1 * (y2 - y1) / 128)), (x1 + int(px2 * (x2 - x1) / 160), y1 + int(py2 * (y2 - y1) / 128)), (255, 0, 0), 2)
            text = f"{model2.names[plate_class_id]}"
            cv2.putText(image, text, (x1 + int(px1 * (x2 - x1) / 160), y1 + int(py1 * (y2 - y1) / 128) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 將檢測結果保存為 JSON 文件
json_path = './picture/detections5.json'
with open(json_path, 'w') as json_file:
    json.dump(detection_results, json_file, indent=4)

# 保存結果圖像
result_image_path = './picture/12.jpg'
cv2.imwrite(result_image_path, image)

print(f"Results saved to {result_image_path}")
print(f"Detection results saved to {json_path}")
print(f"License plates images saved to ./picture/license_plates/")