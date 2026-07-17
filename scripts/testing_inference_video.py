import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO(
    r"C:\Users\ASUS_TUF_I7\OneDrive\Documentos\Docs CV Balsas Defects Project\balsa-defect-poc\exp-5.pt"
)

# Input video
video_path = (
    r"C:\Users\ASUS_TUF_I7\OneDrive\Documentos\Docs CV Balsas Defects Project\balsa-defect-poc\WhatsApp Video 2026-07-14 at 3.48.36 PM.mp4"
)

cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video
output_path = (
    r"C:\Users\ASUS_TUF_I7\OneDrive\Documentos\Docs CV Balsas Defects Project\balsa-defect-poc\output_detection.mp4"
)

fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # Run inference
    results = model(
        frame,
        imgsz=640,
        conf=0.10,
        iou=0.45,
        verbose=False
    )

    # Draw bounding boxes, labels and confidence
    annotated_frame = results[0].plot()

    # Save frame
    out.write(annotated_frame)

    frame_count += 1
    print(f"Processed frame {frame_count}", end="\r")

# Release resources
cap.release()
out.release()

print(f"\nDone! Video saved to:\n{output_path}")