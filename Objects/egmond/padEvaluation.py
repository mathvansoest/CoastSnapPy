
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="pad")
trainer.evaluateModel(model_path="pad/models", json_path="pad/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)