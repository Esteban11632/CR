from ultralytics import YOLO
import os
import sys

def train_model():
    try:
        # Initialize the YOLO model - it will download automatically if not present
        print("Loading/Downloading YOLOv8s model...")
        model = YOLO('yolov8s.pt') # note: changed from yolo8s.pt to yolov8s.pt (correct name)
        print("Model loaded successfully!")

        # Train the model
        print("Starting training...")
        model.train(
            task='detect',
            data=r"C:\Users\DELL\Desktop\Pytorch\RF\CR\Clash royale.v1i.yolov8\data.yaml",
            epochs=1,
            imgsz=768,
            plots=True
        )

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    train_model()

# yolo task=detect mode=train model=yolov8s.pt data="C:\Users\DELL\Desktop\Pytorch\RF\CR\Clash royale.v1i.yolov8\data.yaml" epochs=50 imgsz=768 plots=true