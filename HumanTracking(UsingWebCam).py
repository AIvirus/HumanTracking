import random
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from collections import OrderedDict

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Opening the file in read mode
my_file = open("utils/coco.txt", "r")
# Reading the file
data = my_file.read()
# Replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load the pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8").to(device)

# Vals to resize video frames | small frame optimize the run
frame_wid = 640
frame_hyt = 480

# Initialize centroid tracker
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.zeros((len(object_ids), len(input_centroids)), dtype=int)
            for i, centroid in enumerate(object_centroids):
                for j, coord in enumerate(input_centroids):
                    D[i, j] = np.linalg.norm(np.array(centroid) - np.array(coord))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


# Initialize centroid tracker
centroid_tracker = CentroidTracker()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize the frame while maintaining the aspect ratio
    aspect_ratio = frame.shape[1] / frame.shape[0]
    new_height = int(frame_wid / aspect_ratio)
    frame = cv2.resize(frame, (frame_wid, new_height))

    # Convert the frame to a PIL Image
    pil_image = Image.fromarray(frame)

    # Predict on image
    detect_params = model.predict(source=pil_image, conf=0.45, save=False)

    # Filter out only the detections for the "person" class
    person_detections = []
    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]  # Get one box
        clsID = box.cls.cpu().numpy()[0]  # Get the class ID
        if clsID == 0:  # Check if it corresponds to the "person" class
            person_detections.append(box)

    # Extract bounding boxes for person detections
    rects = []
    for box in person_detections:
        bb = box.xyxy.cpu().numpy()[0]
        startX, startY, endX, endY = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        rects.append((startX, startY, endX, endY))

    # Update centroid tracker with bounding boxes
    objects = centroid_tracker.update(rects)

    # Draw bounding boxes and object ID on the frame
    for (object_id, centroid) in objects.items():
        text = "ID {}".format(object_id)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
