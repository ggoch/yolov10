import cv2
import numpy as np

def crop_image(image_path, coordinates, crop_size=(160, 128)):
    """
    以車牌中心點為中心，裁剪圖片
    """
    # 读取图片
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

    if isinstance(coordinates, str):
        # 如果是字符串，则将其拆分并转换为整数
        coordinates = list(map(int, coordinates.split(',')))

    # 座標格式：(x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = coordinates

    # 計算車牌中心點
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # 計算裁剪區域的半寬和半高
    half_width = crop_size[0] // 2
    half_height = crop_size[1] // 2

    if center_x + half_width > image.shape[1]:
        raise ValueError('裁剪尺寸超出圖片寬度')
    if center_y + half_height > image.shape[0]:
        raise ValueError('裁剪尺寸超出圖片高度')

    # 計算裁切區域的邊界
    crop_x_min = max(center_x - half_width, 0)
    crop_y_min = max(center_y - half_height, 0)
    crop_x_max = min(center_x + half_width, image.shape[1])
    crop_y_max = min(center_y + half_height, image.shape[0])

    # 切割區域
    cropped_image = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    return cropped_image,crop_x_min,crop_y_min

def crop_and_resize_image(image_path, coordinates, output_size=(224, 224)):
    """
    裁剪車牌區域並調整大小
    """
    # 讀取圖片
    image = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),-1)

    if isinstance(coordinates, str):
        # 如果是字符串，则将其拆分并转换为整数
        coordinates = list(map(int, coordinates.split(',')))

    # 座標格式：(x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = coordinates

    x_max,y_min,x_min,y_max = int(x_max),int(y_min),int(x_min),int(y_max)

    # 切割車牌區域
    license_plate = image[y_min:y_max, x_min:x_max]

    # 調整圖片尺寸到指定大小
    resized_license_plate = cv2.resize(license_plate, output_size)

    return resized_license_plate