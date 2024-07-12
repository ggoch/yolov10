from ultralytics import YOLOv10

model_output_path = "./runs/car_no/train"
data_yaml_path = "./data_car.yaml"
model = YOLOv10("yolov10n.pt")

if __name__ == "__main__":
    # yaml裡面的data路徑從datasets開始
    model.train(
        imgsz=192,
        epochs=300,
        data=data_yaml_path,
        batch=16,
        patience=100,
        lr0=0.01,
        lrf=0.1,
        multi_scale=True, 
        # weights="yolov8l.pt",
        # project="runs/detect",
        # name="train",
        # exist_ok=True,
    )
    # model.train(data='coco.yaml', epochs=500, batch=256, imgsz=640)