from ultralytics import YOLO

#model = YOLO('yolo11n.pt')
model = YOLO('yolo11s.pt')
model.train(data='SafetyCheck.yaml', epochs=100)
model.val()
