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

        if cnt % 100 == 0:

            everything_results = real_time_sam.model(
                source=frame,
                device=real_time_sam.device,
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9,
            )

            segmented_objects = process_frame(frame, everything_results[0].masks, yolo_model)

            img = real_time_sam.wrapper_fastsam_prompt(frame, everything_results)

            # Annotate the frame with segmentation and labels
            for mask, label in segmented_objects:
                # Draw the mask as a transparent overlay
                # color = (0, 255, 0)  # Green color for the mask
                overlay = frame.copy()
                frame[mask > 0] = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)[mask > 0]

                # Find the mask centroid for placing the label
                y, x = np.where(mask > 0)
                if len(x) > 0 and len(y) > 0:  # Check if mask has any pixels
                    centroid = (int(x.mean()), int(y.mean()))
                    # Ensure label is a string
                    cv2.putText(img, str(label), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else:
            # skip a frame
            cnt += 1

        cv2.imshow('frame', img)
        # cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
