def flatten_bbox(bbox):
    """
    将嵌套的 tuple 或 list 打平为一维列表
    """
    if isinstance(bbox, (tuple, list)) and len(bbox) == 2:
        return [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
    elif isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        return [bbox[0], bbox[1], bbox[2], bbox[3]]
    raise ValueError("Bounding box format is incorrect")

def calculate_offset(original_plate, crop_coords, current_mark):
    """
    計算標記中心和車牌中心的偏移量
    original_plate = [x_min, y_min, x_max, y_max]
    crop_coords = [crop_x_min, crop_y_min]
    current_mark = [x_min, y_min, x_max, y_max]
    """
    # 原圖中車牌中心
    ox_min, oy_min, ox_max, oy_max = original_plate
    cx_min, cy_min, cx_max, cy_max = current_mark

    plate_center_x = (ox_min + ox_max) / 2
    plate_center_y = (oy_min + oy_max) / 2
    
    # 裁剪後車牌中心
    cropped_plate_center_x = plate_center_x - crop_coords[0]
    cropped_plate_center_y = plate_center_y - crop_coords[1]
    
    # 當前標記中心
    mark_center_x = (cx_min + cx_max) / 2
    mark_center_y = (cy_min + cy_max) / 2
    
    # 計算 offset
    offset_x = cropped_plate_center_x - mark_center_x
    offset_y = cropped_plate_center_y - mark_center_y
    
    return offset_x, offset_y


def calculate_new_position(original_x, original_y, crop_x_min, crop_y_min,offset_x=0,offset_y=0):
    """
    計算在裁剪區域中的相對大小
    """

    new_x = original_x - crop_x_min + offset_x
    new_y = original_y - crop_y_min + offset_y
    
    return new_x, new_y

def calculate_resize_position(output_width, output_height, cropped_width, cropped_height, cropped_x, cropped_y):
    """
    當圖片做resize時，對座標也進行resize
    """
    
    # 计算按比例映射到新图像中的位置
    scale_x = output_width / cropped_width
    scale_y = output_height / cropped_height
    resized_x = cropped_x * scale_x
    resized_y = cropped_y * scale_y
    
    return abs(resized_x), abs(resized_y)

def resize_bounding_box(bbox, crop_x_min, crop_y_min, cropped_width, cropped_height,output_width=160, output_height=128):
    if isinstance(bbox, str):
        # 如果是字符串，则将其拆分并转换为整数
        bbox = list(map(int, bbox.split(',')))
    elif isinstance(bbox, (tuple, list)):
        # 如果是嵌套的 tuple 或 list，则打平
        bbox = flatten_bbox(bbox)
    
    x_min, y_min, x_max, y_max = bbox

    # 计算新的坐标
    new_x_min, new_y_min = calculate_new_position(x_min, y_min, crop_x_min, crop_y_min)
    new_x_max, new_y_max = calculate_new_position(x_max, y_max, crop_x_min, crop_y_min)

    new_x_min,new_y_min = calculate_resize_position(output_width, output_height, cropped_width, cropped_height, new_x_min, new_y_min)
    new_x_max,new_y_max = calculate_resize_position(output_width, output_height, cropped_width, cropped_height, new_x_max, new_y_max)
    
    return new_x_min, new_y_min, new_x_max, new_y_max

def calculate_bounding_box(bbox, crop_x_min, crop_y_min):
    if isinstance(bbox, str):
        # 如果是字符串，则将其拆分并转换为整数
        bbox = list(map(int, bbox.split(',')))
    elif isinstance(bbox, (tuple, list)):
        # 如果是嵌套的 tuple 或 list，则打平
        bbox = flatten_bbox(bbox)
    
    x_min, y_min, x_max, y_max = bbox

    # 计算新的坐标
    new_x_min, new_y_min = calculate_new_position(x_min, y_min, crop_x_min, crop_y_min)
    new_x_max, new_y_max = calculate_new_position(x_max, y_max, crop_x_min, crop_y_min)
    
    return new_x_min, new_y_min, new_x_max, new_y_max