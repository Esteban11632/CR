from ultralytics import YOLO
import os
import sys

def train_model():
    try:
        # Initialize from your previously trained best model
        print("Loading previously trained best model...")
        model = YOLO('runs/detect/train4/weights/best.pt')  # Using your best trained weights
        print("Model loaded successfully!")

        # Continue training
        print("Starting additional training...")
        model.train(
            task='detect',
            data=r"C:\Users\DELL\Desktop\Pytorch\RF\CR\Clash royale.v1i.yolov8\data.yaml",
            epochs=50,  # You might want to add more epochs
            imgsz=768,
            plots=True
        )

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    train_model()