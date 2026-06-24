from ultralytics import YOLO
model = YOLO("models/yolov8n-cls.pt")

model.train(
  data="dataset/images",
  epochs=5,
  imgsz=224,
  save=True,
  save_period=1,
  val=True,
  plots=True,
  augment=True,
  hsv_h=0.015,
  hsv_s=0.7,
  hsv_v=0.4,
  fliplr=0.5,
  flipud=0.0,
  project="runs/classify",
  name="test1",
)
