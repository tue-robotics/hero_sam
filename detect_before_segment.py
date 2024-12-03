import cv2
import numpy as np

from algorithms.fast_sam import fastSamRealTime
from ultralytics import YOLO


def process_frame(frame, masks, yolo_model):
    """Process a single frame: segment objects with FastSAM and label them with YOLO."""
    # Resize the frame to a suitable size for FastSAM (adjust based on your model's input size)
    original_frame = frame.copy()
    # input_size = (1024, 1024)  # Assuming FastSAM requires this input size
    # resized_frame = cv2.resize(frame, input_size)

    segmented_objects = []  # Store labeled objects
    masks = masks.cpu().numpy()

    for mask in masks.data:
        # Resize mask back to the original frame size

        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # Extract ROI (Region of Interest) using the mask
        roi = cv2.bitwise_and(original_frame, original_frame, mask=mask_resized.astype(np.uint8))

        # Save ROI temporarily for YOLO processing
        roi_file = "temp_roi.png"
        cv2.imwrite(roi_file, roi)

        # Perform object recognition with YOLO
        yolo_results = yolo_model(roi_file)

        # Extract label from YOLO results
        label = "Unknown"  # Default label
        for result in yolo_results:
            for box in result.boxes:  # Iterate through detected objects
                class_id = int(box.cls)  # Extract class ID
                label = result.names[class_id]  # Get class name from ID
                break  # Take the first detected object for simplicity

        # Save the label with its corresponding mask
        segmented_objects.append((mask_resized, label))

    return segmented_objects


def main():
    model_size = "large"
    real_time_sam = fastSamRealTime(model_size=model_size)
    yolo_model = YOLO('yolov8m.pt')

    # Initiate camera -> 0:embedded camera of laptop, 1,2,3...: external cams
    cap = cv2.VideoCapture(0)

    # frame counter
    cnt = 0

    while cap.isOpened():

        suc, frame = cap.read()


        """
        1 - convert_frame_to_png
        2 - run_yolo_on_png
        3 - extract_labels_from_yolo_detections
        4 - extract_bbox_from_yolo_detections
        5 - feed_fast_sam_with_bbox_prompt
        """

        # Converting frame to PNG
        frame_copy = frame.copy()
        yolo_frame_file = "yolo_frame.png"
        cv2.imwrite(yolo_frame_file, frame_copy)

        # Running YOLO on PNG
        yolo_results = yolo_model(yolo_frame_file)
        bboxes = yolo_results[0].boxes


        # Feed fastsam with bbox prompt
        everything_results = real_time_sam.model(
                source=frame,
                device=real_time_sam.device,
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9,
            )

        img = real_time_sam.wrapper_fastsam_prompt(frame, everything_results, prompt_type="box", prompt_input=bboxes.xyxy.tolist())

        for i, box in enumerate(bboxes):
            print(box.xyxy)
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            cls_id = int(box.cls[0])
            label = f"{yolo_model.names[cls_id]}: {box.conf[0]:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=3)

        cv2.imshow('frame', img)
        # cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
