import argparse
import time
from pathlib import Path
import multiprocessing
from ultralytics import YOLOv10

import sys

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np


import math

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

from lebel_test import detect,detectV2

from collections import Counter

import shutil

import os
import json
import re
import base64
from PIL import Image
import io
from license_tools.process_img import crop_image,crop_and_resize_image
from license_tools.process_box import calculate_bounding_box,resize_bounding_box,flatten_bbox

def image_to_base64(image_path):
    """将图片文件转换为 Base64 编码的字符串。"""
    with Image.open(image_path) as image:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def convert_to_labelme(data, image_path, image_width, image_height,card_number = None):
    """将车牌识别模型的输出转换为 LabelMe JSON 格式，并包含图片的 Base64 编码。"""
    labelme_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),  # 使用图片文件名
        "imageData": image_to_base64(image_path),   # 图片数据
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    if card_number is not None:
        labelme_data['shapes'].append({
            "label": card_number,
            "points": [
                [0,0]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })

    try:
        # 添加车牌整体和字符的标注
        for item in data["license_plates"]:
            bbox = item["plate"]["bounding_box"]

            if isinstance(bbox, (tuple)):
                bbox = flatten_bbox(bbox)
            if isinstance(bbox, str):
                bbox = list(map(int, bbox.split(',')))

            labelme_data['shapes'].append({
                "label": item["plate"]["label"],
                "points": [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[3]]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })

            for text in item["text_labels"]:
                labelme_data['shapes'].append({
                    "label": text['value'],
                    "points": [
                        text['top_left'][0],
                        text['bottom_right'][1]
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {}
                })

            if item.get('text', '') == '':
                continue

            first_text = item["text_labels"][0]

            for diff in item.get('difference', ''):
                labelme_data['shapes'].append({
                    "label": f'{diff}_*',
                    "points": [
                        first_text['top_left'][0],
                        first_text['bottom_right'][1]
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {}
                })
    except Exception as e:
        print(f"Error processing shapes: {e}")
    
    return labelme_data


def find_difference(left: str, right: str) -> str:
    # 创建两个Counter对象分别统计左边和右边字符串中每个字符的数量
    left_counter = Counter(left)
    right_counter = Counter(right)
    
    # 计算左边字符串中出现次数多于右边字符串的字符
    difference = []
    for char in left_counter:
        if left_counter[char] > right_counter[char]:
            difference.append(char * (left_counter[char] - right_counter[char]))
    
    # 将结果转换为字符串返回
    return ''.join(difference)

def clean_text(text):
    # 使用正則表達式移除非字母數字字符
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return cleaned_text

def custom_sort_key(file_name):
    # 提取文件名中的基名和序号
    match = re.match(r'^(SwipeCard)([+-]?\d*)\.png$', file_name)
    if match:
        base_name = match.group(1)
        number_part = match.group(2)
        
        # 特殊处理SwipeCard.png，使其序号为负无穷大，确保排在最前面
        if number_part == "":
            number = 0
        else:
            number = int(number_part)
        
        # 返回 (基名, 序号, 是否为负数)
        return (base_name, abs(number), -number)
    return (file_name, 0, False)  # 默认排序键

def save_result_to_json_file(path,result):
    with open(path, 'w') as f:
        json.dump(result, f, indent=4)

def process_results(resultList):
    # 提取車牌及其置信度
    license_plates = [item for item in resultList if item.get('label') == 'license plate']
    # 按置信度排序
    license_plates.sort(key=lambda x: x.get('confidence', 0), reverse=True)

    # 取最高的一筆
    # license_plates = license_plates[:1]
    
    # 如果沒有找到車牌，返回None
    if not license_plates:
        return None

    # 轉換車牌位置並存儲
    license_positions = []
    for plate in license_plates:
        box = convert_tensor_to_box(
            (plate['bounding_box'][0], plate['bounding_box'][1]),
            (plate['bounding_box'][2], plate['bounding_box'][3])
        )
        license_positions.append({
            'plate': plate,
            'box': box
        })
    
    # 移除車牌項目，剩下的是其他文本信息
    resultList = [item for item in resultList if item.get('label') != 'license plate']
    
    # 將文本信息轉換成統一格式
    texts = []
    for data in resultList:
        box = convert_tensor_to_box(
            (data['bounding_box'][0], data['bounding_box'][1]),
            (data['bounding_box'][2], data['bounding_box'][3])
        )
        texts.append({'value': data['label'], 'box': box})
    
    # 根據車牌位置分組文本信息
    final_results = []
    for license_data in license_positions:
        plate_box = license_data['box']
        texts_within_plate = [text for text in texts if is_within(plate_box, text['box'])]
        texts_within_plate.sort(key=lambda x: x['box'][0])
        final_text = ''.join(text['value'] for text in texts_within_plate)

        final_texts = [{
            'value': text['value'],
            'top_left': (text['box'][0], text['box'][1]),
            'bottom_right': (text['box'][2], text['box'][3])
            } for text in texts_within_plate]
        
        final_results.append({
            'plate': license_data['plate'],
            'text': final_text,
            'text_labels': final_texts
        })
    
    return final_results

def convert_tensor_to_box(c1,c2):
    # 解包 c1 和 c2 的坐標
    x1, y1 = c1
    x2, y2 = c2

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 計算四個頂點
    top_left = (min(x1, x2), min(y1, y2))
    top_right = (max(x1, x2), min(y1, y2))
    bottom_left = (min(x1, x2), max(y1, y2))
    bottom_right = (max(x1, x2), max(y1, y2))

    return [top_left, top_right, bottom_left, bottom_right]

def is_within(plate_box, text_box):
    plate_top_left, plate_top_right, plate_bottom_left, plate_bottom_right = plate_box
    text_top_left, text_top_right, text_bottom_left, text_bottom_right = text_box

    def is_point_within_plate(point):
        x, y = point
        # 車牌的 x 和 y 坐標範圍
        min_x = min(plate_top_left[0], plate_bottom_left[0])
        max_x = max(plate_top_right[0], plate_bottom_right[0])
        min_y = min(plate_top_left[1], plate_top_right[1])
        max_y = max(plate_bottom_left[1], plate_bottom_right[1])

        return min_x <= x <= max_x and min_y <= y <= max_y

    return (is_point_within_plate(text_top_left) and
            is_point_within_plate(text_top_right) and
            is_point_within_plate(text_bottom_left) and
            is_point_within_plate(text_bottom_right))

def check_is_predicted(dir_path, threshold=3):
    folder_count = 0
    for file in os.listdir(dir_path):
        full_path = os.path.join(dir_path, file)
        if os.path.isdir(full_path):
            folder_count += 1
            if folder_count >= threshold:
                print(f"Skipping directory: {full_path}")
                return True
    return False

def parse_files(dir_path):
    plus_files = []
    minus_files = []
    base_file = None

    for file in os.listdir(dir_path):
        if file.startswith('SwipeCard') and file.endswith(('.png', '.jpg', '.jpeg')):
            match = re.match(r'SwipeCard([+-]?\d*)\.(png|jpg|jpeg)', file, re.IGNORECASE)
            if match:
                num_part = match.group(1)
                if num_part.startswith('+'):
                    plus_files.append(int(num_part[1:]))
                elif num_part.startswith('-'):
                    minus_files.append(int(num_part))
                else:
                    base_file = file

    return plus_files, minus_files, base_file

def find_extremes(plus_files, minus_files):
    max_plus = max(plus_files) if plus_files else None
    min_minus = min(minus_files) if minus_files else None
    return max_plus, min_minus

def find_next_valid_file(dir_path,output_path, prefix, start_num, step=1,break_num=None,filter_nums=[],count=1,license_max_count=1,model1=None,model2=None):
    num = start_num
    file_names = []

    if count == 0:
        return None

    while True:
        if break_num is not None and num == break_num:
            if len(file_names) == 0:
                return None
            else:
                return file_names
        
        if num in filter_nums:
            num += step
            continue
        
        for ext in ['.png', '.jpg', '.jpeg']:
            file_name = f'{prefix}{num}{ext}'
            file_path = os.path.join(dir_path, file_name)
            if os.path.exists(file_path):
                if detect_img(file_path,output_path,license_max_count,model1,model2):
                    file_names.append(file_name)
                    if len(file_names) >= count:
                        return file_names
        num += step

def detect_img(image_path,save_dir='results',license_max_count=1,model1=None,model2=None):
    try:
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):

            image_name = os.path.basename(image_path)
            event_id = os.path.basename(os.path.dirname(image_path))
            car_no = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
            name = os.path.splitext(image_name)[0]
            # predicted_card_result = detect(image_path, r"Z:\Yolov7\Yolov7\yolov7\licenseplate.pt",project=id_path,name=name)
            # predicted_card_result = detect(image_path, "./licenseplate.pt",project=id_path,name=name)

            # if hasattr(sys, '_MEIPASS'):
            #     model_path = os.path.join(sys._MEIPASS, 'licenseplate.pt')
            # else:
            #     model_path = './licenseplate.pt'

            # predicted_card_result = detect(image_path, model_path,name=name,nosave=True)
            # predicted_card_result = detect(image_path,license_max_count,model1,model2)
            predicted_card_result = detectV2(image_path,license_max_count,model1)

            # result_img_path = os.path.join(id_path,name)

            if predicted_card_result is None:
                print(f"No License Plate Detected: {image_path}")
                # save_result_to_json_file(os.path.join(save_dir, f"{name}.json"), {'license_plates': None})
                return False

            all_texts_empty = True

            for license_plate in predicted_card_result['license_plates']:
                if len(license_plate['text']) != 0:
                    all_texts_empty = False
                    break

            if all_texts_empty:
                print(f"No License Plate Detected: {image_path}")
                return False                                    

            # 整理車牌模型的輸出可Json化    
            for license_plate in predicted_card_result['license_plates']:
                license_plate['result'] = license_plate['text'] == clean_text(car_no)

                for texts in license_plate['text_labels']:
                    for top_left in texts['top_left']:
                        top_left = map(int, top_left)
                    for bottom_right in texts['bottom_right']:
                        bottom_right = map(int, bottom_right)

                bounding_box = license_plate['plate']['bounding_box']
                license_plate['plate']['bounding_box'] = [int(coord) for coord in bounding_box]

                license_plate['plate']['confidence'] = float(license_plate['plate']['confidence'])

                if license_plate['result']:
                    print(f"Match: {image_path} -> {license_plate['text']}")
                    license_plate['difference'] = ''
                else:
                    license_plate['difference'] = find_difference(clean_text(car_no),license_plate['text'])
                    print(f"No Match: {image_path} -> {license_plate['text']} (Expected: {car_no})")

            # for item in predicted_card_result['resultList']:
            #     item['bounding_box'] = f"{int(item['bounding_box'][0])}, {int(item['bounding_box'][1])}, {int(item['bounding_box'][2])}, {int(item['bounding_box'][3])}"
            #     item['confidence'] = float(item['confidence'])

            # labelme_json = convert_to_labelme(predicted_card_result, image_path, 1920, 1080,clean_text(car_no))

            # save_dir = os.path.join(id_path,name)
            save_dir = os.path.join(save_dir,car_no)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # base_image = cv2.imread(image_path)
            # cv2.imwrite(os.path.join(save_dir, f"{name}.jpg"), base_image)

            # 保存结果到json文件
            # save_result_to_json_file(os.path.join(save_dir, f"{name}_result.json"), predicted_card_result)
            # save_result_to_json_file(os.path.join(save_dir, f"{name}.json"), labelme_json)

            for idx, license_plate in enumerate(predicted_card_result['license_plates']):
                # card_image = crop_and_resize_image(image_path, license_plate['plate']['bounding_box'], output_size=(160, 128))
                try:
                    car_image,crop_x_min,crop_y_min = crop_image(image_path, license_plate['plate']['bounding_box'], crop_size=(160, 128))
                except ValueError as e:
                    print(f"該圖車牌超出預設大小待人員確認: {image_path}, {e}")
                    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
                    save_image_path = os.path.join(save_dir, f"{event_id}_{name}_{idx}_over_crop.jpg")
                    result, encoded_img = cv2.imencode(".jpg", image)
                    encoded_img.tofile(save_image_path)
                    continue

                license_plate_resize = {
                    "plate": license_plate['plate'],
                    "text_labels": license_plate['text_labels'],
                    "difference": license_plate['difference'],
                    "text": license_plate['text'],
                    # "difference": find_difference(clean_text(card_number_dir),license_plate['text'])
                }
                # 須對圖片座標做resize才需要
                # original_width, original_height = 1920, 1080
                # new_width, new_height = 160, 128

                # license_plate_resize['plate']['bounding_box'] = [int(coord) for coord in license_plate_resize['plate']['bounding_box'].split(',')]

                # crop_x, crop_y = license_plate['plate']['bounding_box'][0], license_plate['plate']['bounding_box'][1]
                # crop_width = license_plate_resize['plate']['bounding_box'][0] - license_plate_resize['plate']['bounding_box'][2]
                # crop_height = license_plate_resize['plate']['bounding_box'][1] - license_plate_resize['plate']['bounding_box'][3]

                # license_plate_resize['plate']['bounding_box'] = resize_bounding_box(license_plate_resize['plate']['bounding_box'], crop_x, crop_y,crop_width, crop_height, new_width, new_height)
                license_plate_resize['plate']['bounding_box'] = calculate_bounding_box(license_plate_resize['plate']['bounding_box'], crop_x_min, crop_y_min)

                for text in license_plate_resize["text_labels"]:
                    # top_left = resize_bounding_box(text['top_left'], crop_x, crop_y,crop_width, crop_height, new_width, new_height)
                    top_left = calculate_bounding_box(text['top_left'], crop_x_min, crop_y_min)
                    # bottom_right = resize_bounding_box(text['bottom_right'], crop_x, crop_y,crop_width, crop_height, new_width, new_height)
                    bottom_right = calculate_bounding_box(text['bottom_right'], crop_x_min, crop_y_min)

                    text['top_left']= ((top_left[0],top_left[1]),(top_left[2],top_left[3]))
                    text['bottom_right']= ((bottom_right[0],bottom_right[1]),(bottom_right[2],bottom_right[3]))


                # save_image_path = os.path.join(save_dir, f"{name}_{idx}_license.jpg")
                save_image_path = os.path.join(save_dir, f"{event_id}_{name}_{idx}_license.jpg")
                result, encoded_img = cv2.imencode(".jpg", car_image)

                # if cv2.imwrite(save_image_path, card_image):
                if result:
                    if os.path.exists(save_image_path):
                        print(f"檔案 {save_image_path} 已存在，避免覆寫。")
                        continue
                    
                    encoded_img.tofile(save_image_path)
                    print("save success", save_image_path)
                    labelme_json = convert_to_labelme({
                        'license_plates': [license_plate_resize],
                        'resultList': []
                    }, save_image_path, 160, 128,clean_text(car_no))
                    save_result_to_json_file(os.path.join(save_dir, f"{event_id}_{name}_{idx}_license.json"), labelme_json)
                else:
                    print("save error", save_image_path)
                    continue

            return True
        else:
            return False
    except Exception as e:
            print(f"Error processing image: {image_path}, {e}")
            return False


def get_image_frame_list(file_names):
    process_frame_list = []

    for file in file_names:
        # 检查文件名中是否包含 '+' 或 '-'
        if '+' in file:
            parts = file.split('+')
            if len(parts) > 1:
                number_part = parts[1].split('.')[0]
        elif '-' in file:
            parts = file.split('-')
            if len(parts) > 1:
                number_part = '-' + parts[1].split('.')[0]
        else:
            continue  # 如果文件名不包含 '+' 或 '-'，跳过此文件

        # 确保分割后的结果长度足够
        try:
            process_frame_list.append(int(number_part))
        except ValueError:
            print(f"Error: Cannot convert {number_part} to int")

    return process_frame_list

def process_images(folder_path,output_path="//192.168.1.10/113-智慧過磅暨ai影像追蹤系統研發計畫/0626/Label",count=3,license_max_count=1,model1=None,model2=None):
    """
    根據目錄中的圖片進行車牌辨識，預設三個時間，分別為刷卡帧、影片結束前7帧、進磅帧
    """
    folder_path = folder_path.replace('\\', '/')

    plus_files, minus_files, base_file = parse_files(folder_path)
    max_plus, min_minus = find_extremes(plus_files, minus_files)

    name, extension = os.path.splitext(base_file)

    base_file_path = os.path.join(folder_path, base_file)

    result_plus_path = os.path.join(folder_path, f"SwipeCard+{max_plus - 7}{extension}")

    result_minus_path = os.path.join(folder_path, f"SwipeCard{min_minus}{extension}")

    base_file_result = detect_img(base_file_path,output_path,license_max_count,model1,model2)

    result_plus_result = False

    result_minus_result = False

    result_fileName = base_file if base_file_result else None
    result_plus_fileNames = None
    result_minus_fileNames = None

    # 原始算法，平均分配
    # plus_count = math.ceil((count - 1) / 2)
    # minus_count = (count-1)-plus_count

    # 調整算法，負優先
    plus_count = 0
    minus_count = (count-1)

    process_frame_list = []

    if base_file_result == False:
        result_fileName = find_next_valid_file(folder_path,output_path, 'SwipeCard+', 1, step=1, break_num=max_plus,license_max_count=license_max_count,filter_nums=process_frame_list,model1=model1,model2=model2)
        if result_fileName is not None:
            result_fileName = result_fileName[0]
            # process_frame_list.append(int(result_fileName.split('+')[1].split('.')[0]))
            base_file_result = True

    if min_minus is not None:
        result_minus_fileNames = find_next_valid_file(folder_path,output_path, 'SwipeCard', min_minus,step=1,break_num=max_plus,license_max_count=license_max_count,filter_nums=process_frame_list,count=minus_count,model1=model1,model2=model2)
        if result_minus_fileNames is not None and len(result_minus_fileNames) > 0:
            # process_frame_list += [int(file.split('+')[1].split('.')[0]) for file in result_minus_fileNames]
            process_frame_list += get_image_frame_list(result_minus_fileNames)
            result_minus_result = True

    plus_count = max(plus_count, count - len(process_frame_list))

    if max_plus is not None:
        result_plus_fileNames = find_next_valid_file(folder_path,output_path, 'SwipeCard+', max_plus - 7, step=-1, break_num=min_minus,license_max_count=license_max_count,filter_nums=process_frame_list,count=plus_count,model1=model1,model2=model2)
        if result_plus_fileNames is not None and len(result_plus_fileNames) > 0:
            process_frame_list += get_image_frame_list(result_plus_fileNames)
            result_plus_result = True

    # result_plus_result = detect_img(result_plus_path,output_path)

    # result_minus_result = detect_img(result_minus_path,output_path)

    if base_file_result:
        print(f"刷卡帧保存成功")
    else:
        print(f"刷卡帧找不到車牌")

    if result_plus_result:
        print(f"影片結束前7帧保存成功")
    else:
        print(f"影片結束前7帧找不到車牌")

    if result_minus_result:
        print(f"進磅帧保存成功")
    else:
        print(f"進磅帧找不到車牌")

    return {
        'base_file': {
            "is_success": base_file_result,
            "file_name": result_fileName
        },
        'result_plus': {
            "is_success": result_plus_result,
            "file_names": result_plus_fileNames
        },
        'result_minus': {
            "is_success": result_minus_result,
            "file_names": result_minus_fileNames
        }
    }

def browse_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, folder_selected)

def on_submit(treeview,model1,model2):
    def inner():
        path = entry_path.get().strip()
        path = path.replace('\\', '/')

        _output_path = output_path.get().strip()
        _output_path = _output_path.replace('\\', '/')

        _find_count = find_count.get().strip()
        _find_count = int(_find_count) if _find_count.isdigit() else 3

        _license_max_count = license_max_count.get().strip()
        _license_max_count = int(_license_max_count) if _license_max_count.isdigit() else 1

        # 清除现有数据
        for item in treeview.get_children():
            treeview.delete(item)

        if os.path.isdir(path) and os.path.exists(path):
            result = process_images(path,_output_path,_find_count,_license_max_count,model1,model2)

            table_data = []

            messageText = ""

            if result['base_file']["is_success"]:
                messageText += f"刷卡帧保存成功 : {result['base_file']['file_name']}\n"
                table_data.append(("刷卡帧",result['base_file']['file_name']))
            else:
                messageText += "刷卡帧找不到車牌\n"
                table_data.append(("刷卡帧","沒有找到車牌"))

            if result['result_plus']["is_success"]:
                for file in result['result_plus']['file_names']:
                    messageText += f"影片結束前7帧保存成功 : {file}\n"
                    table_data.append(("影片結束前7帧",file))
            else:
                messageText += "影片結束前7帧找不到車牌\n"

            if result['result_minus']["is_success"]:
                for file in result['result_minus']['file_names']:
                    messageText += f"進磅帧保存成功 : {file}\n"
                    table_data.append(("進磅帧",file))
            else:
                messageText += "進磅帧找不到車牌\n"

            reader_message_table(table_data,treeview)

            messagebox.showinfo("偵測結束", messageText)
        else:
            messagebox.showerror("錯誤", "路徑不存在。請選擇有效路徑")
    
    return inner

def reader_message_table(data,treeview):
    columns = ("Time", "Name")

    treeview.heading("Time", text="找尋開始時間")
    treeview.heading("Name", text="名稱")

    treeview.column("Time", width=100)
    treeview.column("Name", width=200)

    for row in data:
        treeview.insert("", tk.END, values=row)

    # treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    treeview.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def on_copy(event):
    selected_items = treeview.selection()
    if selected_items:
        data_to_copy = []
        for item in selected_items:
            row_data = treeview.item(item, 'values')
            # data_to_copy.append('\t'.join(row_data))
            data_to_copy.append(row_data[1])
        # data_to_copy.append('\t'.join(treeview.item("Name", 'values')))
        clipboard_data = '\n'.join(data_to_copy)
        root.clipboard_clear()
        root.clipboard_append(clipboard_data)
        print(f"Copied to clipboard: \n{clipboard_data}")


# process_images_in_directory("./yolov7/CarNo")
#process_images_in_directory_new(r"\\192.168.1.10\113-智慧過磅暨ai影像追蹤系統研發計畫\攝影機影像\random_fff")
# process_images_in_directory_new("../../../random_fff")
    # process_images_in_directory_new("../../../攝影機影像/text7-1")
    # detect_img("./CarNo/001-T8/19C9C3F8-8E58-08B3-6836-3A0AC84EB00B/SwipeCard.png")
    # process_images(r"./CarNo/001-T8/19C9C3F8-8E58-08B3-6836-3A0AC84EB00B")
    # process_images(r"\\192.168.1.10\113-智慧過磅暨ai影像追蹤系統研發計畫\攝影機影像\text7-1\KEB6012\303F79FC-365A-E872-9DEC-3A0CF1EA2D81")
if __name__ == "__main__":
    # process_images(r"./CarNo/001-T8/19C9C3F8-8E58-08B3-6836-3A0AC84EB00B",'./results')
    # process_images(r"\\192.168.1.10\113-智慧過磅暨ai影像追蹤系統研發計畫-V1\攝影機影像\text7-1\131BR\00EDE170-C9D8-2FAC-B621-3A0CC2151DD9",'./results')

    # parser = argparse.ArgumentParser(description='請輸入資料夾位置，會自動分析資料夾內的圖片')
    # parser.add_argument('path', type=str, help='請輸入資料夾位置')

    # args = parser.parse_args()
    # process_images(args.path)

    # Create the main window
    multiprocessing.freeze_support()
    if hasattr(sys, '_MEIPASS'):
        # model1_path = os.path.join(sys._MEIPASS, 'licensr_position.pt')
        model1_path = os.path.join(sys._MEIPASS, 'license_car_noV2.pt')
    else:
        # model1_path = './licensr_position.pt'
        model1_path = './license_car_noV2.pt'
        # model1_path = './runs/detect/train78/weights/best.pt'
    # 加載第一次檢測的YOLO模型
    model1 = YOLOv10(model1_path)

    if hasattr(sys, '_MEIPASS'):
        model2_path = os.path.join(sys._MEIPASS, 'car_no.pt')
    else:
        model2_path = './car_no.pt'
    # 加載第二次檢測的YOLO模型
    model2 = YOLOv10(model2_path)

    root = tk.Tk()
    root.title("偵測車牌影像")

    # Create and place the widgets
    label_path = tk.Label(root, text="選擇資料夾路徑:")
    label_path.pack(pady=10)

    entry_path = tk.Entry(root, width=80)
    entry_path.pack(padx=20, pady=5)
    # entry_path.insert(0, "./CarNo/001-T8/19C9C3F8-8E58-08B3-6836-3A0AC84EB00B")  # 设置初始值

    output_path = tk.Entry(root, width=80)
    output_path.pack(padx=20, pady=5)
    output_path.insert(0, "//192.168.1.10/113-智慧過磅暨ai影像追蹤系統研發計畫-V1/0626/Label")  # 设置初始值
    # output_path.insert(0, "./result")  # 设置初始值

    find_count = tk.Entry(root, width=80)
    find_count.pack(padx=20, pady=5)
    find_count.insert(0, "3")  # 设置初始值

    license_max_count = tk.Entry(root, width=80)
    license_max_count.pack(padx=20, pady=5)
    license_max_count.insert(0, "3")  # 设置初始值

    # btn_browse = tk.Button(root, text="瀏覽", command=browse_folder)
    # btn_browse.pack(pady=5)

    columns = ("Time", "Name")
    treeview = ttk.Treeview(root, columns=columns, show='headings')

    treeview.bind("<Control-c>", on_copy)

    btn_submit = tk.Button(root, text="偵測", command=on_submit(treeview,model1,model2))
    btn_submit.pack(pady=20)

    # Run the application
    root.mainloop()
