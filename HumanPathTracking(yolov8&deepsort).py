from collections import defaultdict
import cv2
import numpy as np
import torch
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Load DeepSORT
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

# Define the video path
video_path = "videos\\street.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video capture object is opened successfully
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Store the track history and colors for each track
track_history = defaultdict(lambda: [])
track_colors = {}

# Set your screen resolution
screen_width = 1280
screen_height = 720

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (screen_width, screen_height))

# Define the size of the display window
window_width = 800
window_height = 600

# Function to generate random colors
def generate_colors(num_colors):
    return np.random.randint(0, 255, size=(num_colors, 3))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        yolo_results = yolo_model.track(frame, persist=True, classes=0)

        # Get the boxes and track IDs from YOLOv8
        boxes = yolo_results[0].boxes.xywh.cpu()
        track_ids = yolo_results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = yolo_results[0].plot()

        # Update the DeepSORT tracker with the detections
        deep_sort_boxes = boxes.cpu().numpy()
        confidences = yolo_results[0].boxes.conf.cpu().numpy()
        tracks = tracker.update(deep_sort_boxes, confidences, frame)

        # Plot the tracks
        for track in tracks:
            track_id = track[4]  # Assuming track_id is at index 4, adjust as per your track representation
            x1, y1, x2, y2 = track[:4]  # Extract bounding box coordinates

            # Calculate width and height
            w = x2 - x1
            h = y2 - y1

            # Generate a color for the track if it doesn't have one
            if track_id not in track_colors:
                track_colors[track_id] = tuple(np.random.randint(0, 255, size=3).tolist())

            # Draw the bounding box and track ID with the assigned color
            color = track_colors[track_id]
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            cv2.putText(annotated_frame, f"Track ID: {track_id}", (int(x1) + 10, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Update track history
            track_history[track_id].append((int(x1 + w / 2), int(y1 + h / 2)))  # Update with center point

            # Draw the track history
            if len(track_history[track_id]) > 1:
                for i in range(1, len(track_history[track_id])):
                    cv2.line(annotated_frame, track_history[track_id][i - 1], track_history[track_id][i],
                             color, 2)

        # Resize the frame to fit the specified window size
        resized_frame = cv2.resize(annotated_frame, (window_width, window_height))

        # Write the annotated frame to the output video
        out.write(resized_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 + DeepSORT Tracking", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
