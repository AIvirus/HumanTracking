from ultralytics import YOLO
import time
import torch
import cv2
import numpy as np
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

# Load YOLO model
model = YOLO("yolov8n.pt")

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Load DeepSORT
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

# Define the Real Time video path
cap = cv2.VideoCapture(0)
address = "http://______________/video"       #In place of '_______' write https of your webcam ( I used IP WEBCAM app from playstore to create that)
cap.open(address)

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

frames = []
unique_track_ids = set()
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()

# Set your screen resolution
screen_width = 1280
screen_height = 720

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = og_frame.copy()

        # Resize the frame to fit within the screen resolution
        resized_frame = cv2.resize(og_frame, (screen_width, screen_height))

        # Perform object detection
        results = model(resized_frame, device='cuda', classes=0, conf=0.5)

        for result in results:
            boxes = result.boxes
            conf = boxes.conf.detach().cpu().numpy()
            xyxy = boxes.xyxy.detach().cpu().numpy()
            xywh = boxes.xywh.detach().cpu().numpy()

            for i in range(len(boxes)):
                class_index = int(boxes.cls[i])
                class_name = class_names[class_index]

        bboxes_xywh = xywh
        tracks = tracker.update(bboxes_xywh, conf, og_frame)

        for track in tracks:
            track_id = track[4]  # Assuming track_id is at index 4, adjust as per your track representation
            x1, y1, x2, y2 = track[:4]  # Extract bounding box coordinates

            # Calculate width and height
            w = x2 - x1
            h = y2 - y1

            color = (0, 255, 0)  # Green color
            cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            cv2.putText(resized_frame, f"Track ID: {track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            unique_track_ids.add(track_id)

        person_count = len(unique_track_ids)

        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        cv2.putText(resized_frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)

        frames.append(resized_frame)
        out.write(cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))

        # Show the resized frame
        cv2.imshow("Video", cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
