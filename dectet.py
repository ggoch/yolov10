from ultralytics import YOLO,YOLOv10
import cv2
from boxs.process_predict_result import get_car_no_and_license_plates_by_result
from lebel_test import detectV2

image_path = 'data/SwipeCard03.jpg'  # Image folder
# image_path = 'data/0F8C3649-E1D5-02D1-7E6F-3A0EB9B5DDAD_SwipeCard_0_license.jpg'  # Image folder

model = YOLOv10('./runs/detect/train77/weights/best.pt')  # Load model

# results = model.predict(image_path,imgsz=1024)  # Run inference

result = detectV2(image_path,1,model)

print(result)

# result_boxes = get_car_no_and_license_plates_by_result(results[0],model.names)  # Get license plates

# img = results[0].plot()  # Get image

# cv2.imshow('Image', img)  # Display image
# cv2.waitKey(0)  # Wait for keypress
# cv2.destroyAllWindows()  # Close window