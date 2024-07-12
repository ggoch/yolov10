from ultralytics import YOLOv10
import cv2
import json
import os
from boxs.process_predict_result import get_license_plates_by_result,get_car_no_by_result,get_car_no_and_license_plates_by_result
from convert.process_txt import process_result_result_car_no
import sys
import numpy as np

def detect(image_path,license_max_count=1,input_model1=None,input_model2=None):
    if hasattr(sys, '_MEIPASS'):
        model1_path = os.path.join(sys._MEIPASS, 'licensr_position.pt')
    else:
        model1_path = './licensr_position.pt'
    # 加載第一次檢測的YOLO模型
    model1 = YOLOv10(model1_path) if input_model1 is None else input_model1

    if hasattr(sys, '_MEIPASS'):
        model2_path = os.path.join(sys._MEIPASS, 'car_no.pt')
    else:
        model2_path = './car_no.pt'
    # 加載第二次檢測的YOLO模型
    model2 = YOLOv10(model2_path) if input_model2 is None else input_model2

    # 加載圖像
    # image_path = r'./SwipeCard.jpg'
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

    # 使用第一次檢測模型進行檢測
    results1 = model1.predict(image, imgsz=640)

    license_plates,detection_results = get_license_plates_by_result(results1, model1.names)

    convert_result = []

    for idx, (x1, y1, x2, y2) in enumerate(license_plates):
        license_plate_image = image[y1:y2, x1:x2]
        resized_license_plate_image = cv2.resize(license_plate_image, (160, 128))
        # license_plate_image_path = f'./picture/license_plates/license_plate_{idx}.jpg'
        # cv2.imwrite(license_plate_image_path, resized_license_plate_image)

        plate_results = model2.predict(resized_license_plate_image, imgsz=160)

        car_no_result = get_car_no_by_result(plate_results,model1.names,(x1, y1, x2, y2))

        car_no,filtered_boxes,license_plate = process_result_result_car_no(car_no_result['detections'],model1.names)

        print(f"{idx}的車牌號碼為{car_no}")

        convert_result.append({
            'plate': {
                'label': license_plate[4],
                'confidence': license_plate[5],
                'bounding_box': [license_plate[0], license_plate[1], license_plate[2], license_plate[3]]
            },
            'text': car_no,
            'text_labels': list(map(lambda x: {
                'value': x[4],
                'top_left': ((x[0], x[1]), (x[2], x[1])),
                'bottom_right': ((x[0], x[3]), (x[2], x[3]))
            }, filtered_boxes))
        })

        print(convert_result)

    convert_result.sort(key=lambda x: x['plate']['confidence'],reverse=True)

    convert_result = convert_result[:license_max_count]

    return {
        'license_plates':convert_result
    }

def detectV2(image_path,license_max_count=1,input_model=None):
    if hasattr(sys, '_MEIPASS'):
        model_path = os.path.join(sys._MEIPASS, 'license_car_no.pt')
    else:
        model_path = './license_car_no.pt'
    # 加載第一次檢測的YOLO模型
    model = YOLOv10(model_path) if input_model is None else input_model

    # 加載圖像
    # image_path = r'./SwipeCard.jpg'
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

    results = model.predict(image_path,imgsz=1024)  # Run inference

    result_boxes = get_car_no_and_license_plates_by_result(results[0],model.names)  # Get license plates

    license_plates = []
    texts = []


    # 遍歷檢測結果，分類車牌和文字
    for box in result_boxes['detections']:
        x1, y1, x2, y2 = box['bbox']  # 獲取邊界框座標
        cls = box['class_name']  # 獲取類別

        if cls == 'license plate':  # 假設車牌類別標籤是'license_plate'
            license_plates.append(box)
        else:  # 假設文字類別標籤是'text'
            texts.append(box)

    # 過濾掉沒有文字的車牌
    convert_result = []

    for lp in license_plates:
        inner_texts = []
        lp_x1, lp_y1, lp_x2, lp_y2 = lp['bbox']
        has_text = False
        for txt in texts:
            txt_x1, txt_y1, txt_x2, txt_y2 = txt['bbox']
            # 檢查文字框是否在車牌框內
            if txt_x1 >= lp_x1 and txt_y1 >= lp_y1 and txt_x2 <= lp_x2 and txt_y2 <= lp_y2:
                has_text = True
                inner_texts.append(txt)

        if has_text:
            car_no,filtered_boxes,license_plate = process_result_result_car_no(inner_texts,model.names)
            convert_result.append({
            'plate': {
                'label': lp['class_name'],
                'confidence': lp['confidence'],
                'bounding_box': lp['bbox']
            },
            'text': car_no,
            'text_labels': list(map(lambda x: {
                'value': x[4],
                'top_left': ((x[0], x[1]), (x[2], x[1])),
                'bottom_right': ((x[0], x[3]), (x[2], x[3]))
            }, filtered_boxes))
        })
            
    convert_result.sort(key=lambda x: x['plate']['confidence'],reverse=True)

    convert_result = convert_result[:license_max_count]

    return {
        'license_plates':convert_result
    }