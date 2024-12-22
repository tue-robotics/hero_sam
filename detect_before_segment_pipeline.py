import cv2

from algorithms.fast_sam import fastSamRealTime
from ultralytics import YOLO


def capture_frame(cap):
    """Capture a frame from the video source."""
    success, frame = cap.read()
    if not success:
        raise RuntimeError("Failed to read frame from video capture.")
    return frame


def save_frame_as_png(frame, filename):
    """Save a frame as a PNG file."""
    cv2.imwrite(filename, frame)
    return filename


def run_yolo_on_frame(frame_file, yolo_model):
    """Run YOLO object detection on the given frame file."""
    results = yolo_model(frame_file)
    return results[0].boxes


def feed_fast_sam_with_bbox(frame, bboxes, sam_model):
    """Run FastSAM on the frame using bounding box prompts."""
    everything_results = sam_model.model(
        source=frame,
        device=sam_model.device,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )

    return sam_model.wrapper_fastsam_prompt(
        frame,
        everything_results,
        prompt_type="box",
        prompt_input=bboxes.xyxy.tolist()
    )


def annotate_frame_with_labels(frame, bboxes, yolo_model):
    """Annotate the frame with labels and bounding boxes."""
    for box in bboxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        cls_id = int(box.cls[0])
        label = f"{yolo_model.names[cls_id]}: {box.conf[0]:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color=3
        )
    return frame


def create_pipeline():
    """Set up and execute the video processing pipeline."""
    # Initialize models
    model_size = "large"
    real_time_sam = fastSamRealTime(model_size=model_size)
    yolo_model = YOLO('yolov8m.pt')

    # Set up video capture
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            # Step 1: Capture a frame
            frame = capture_frame(cap)

            # Step 2: Save frame as PNG for YOLO
            yolo_frame_file = save_frame_as_png(frame, "yolo_frame.png")

            # Step 3: Run YOLO on the frame
            bboxes = run_yolo_on_frame(yolo_frame_file, yolo_model)

            # Step 4: Feed FastSAM with YOLO bounding box prompts
            processed_frame = feed_fast_sam_with_bbox(frame, bboxes, real_time_sam)

            # Step 5: Annotate the frame with YOLO labels
            annotated_frame = annotate_frame_with_labels(processed_frame, bboxes, yolo_model)

            # Display the processed frame
            cv2.imshow('Processed Frame', annotated_frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    create_pipeline()
