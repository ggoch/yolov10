import cv2

def get_license_plates_by_result(results,names):
    """
    分析車牌檢測結果，找尋車牌位置
    results: 檢測結果
    names: 模型類別名稱
    """
    # 繪製第一次檢測結果並存儲到字典中
    license_plates = []

    # 準備存儲檢測結果的字典
    detection_results = {
        # 'image_path': image_path,
        'detections': []
    }

    for result in results:
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
                'class_name': names[class_id],
                'confidence': score
            })
            # 如果檢測到的類別是車牌，則記錄車牌區域
            if names[class_id] == 'license plate':
                license_plates.append((x1, y1, x2, y2))
            # 在圖片上繪製第一次檢測的框和類別名稱
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # text = f"{names[class_id]}"
            # cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return license_plates,detection_results

def get_car_no_by_result(results,names,license_plates):
    # 準備存儲檢測結果的字典
    detection_results = {
        # 'image_path': image_path,
        'detections': []
    }

    x1, y1, x2, y2 = license_plates

    for plate_result in results:
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
                'class_name': names[plate_class_id],
                'confidence': plate_score
            })
            # 在原圖上繪製第二次檢測的框和類別名稱
            # cv2.rectangle(image, (x1 + int(px1 * (x2 - x1) / 160), y1 + int(py1 * (y2 - y1) / 128)), (x1 + int(px2 * (x2 - x1) / 160), y1 + int(py2 * (y2 - y1) / 128)), (255, 0, 0), 2)
            # text = f"{names[plate_class_id]}"
            # cv2.putText(image, text, (x1 + int(px1 * (x2 - x1) / 160), y1 + int(py1 * (y2 - y1) / 128) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return detection_results

def get_car_no_and_license_plates_by_result(results,names):
    # 準備存儲檢測結果的字典
    detection_results = {
        # 'image_path': image_path,
        'detections': []
    }

    for plate_result in results:
        plate_boxes = plate_result.boxes.xyxy
        plate_class_ids = plate_result.boxes.cls
        plate_scores = plate_result.boxes.conf
        for plate_box, plate_class_id, plate_score in zip(plate_boxes, plate_class_ids, plate_scores):
            px1, py1, px2, py2 = plate_box.int().tolist()
            plate_class_id = plate_class_id.int().item()
            plate_score = plate_score.item()
            detection_results['detections'].append({
                'bbox': [px1, py1, px2, py2],
                'class_id': plate_class_id,
                'class_name': names[plate_class_id],
                'confidence': plate_score
            })

    return detection_results