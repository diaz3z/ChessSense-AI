import cv2
import streamlit as st
from ultralytics import YOLO
import supervision as sv

# Define the frame width and height for video capture
frame_width = 1280
frame_height = 720

def main():
    # Set page title and header
    st.title("Live Object Detection with YOLOv8")

    # Button to start the camera
    start_camera = st.button("Start Camera")

    if start_camera:
        # Initialize video capture from default camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        # Load YOLOv8 model
        model = YOLO("runs/detect/train6/weights/best.pt")

        # Initialize box annotator for visualization
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

        # Main loop for video processing
        while True:
            # Read frame from video capture
            ret, frame = cap.read()

            # Perform object detection using YOLOv8
            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)

            # Prepare labels for detected objects
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]

            # Annotate frame with bounding boxes and labels
            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )

            # Display annotated frame
            st.image(frame, channels="BGR", use_column_width=True)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture
        cap.release()

if __name__ == "__main__":
    main()