import cv2
import torch
import numpy as np
from fastsam import FastSAM, FastSAMPrompt

import time
import threading


##

track_target = 'rabbit'

fps = 3
frame_diff_epsilon = 10 # Threshold for frame difference
lock = threading.Lock() # FPS auto throttle

cap = cv2.VideoCapture(0)
sam = FastSAM("FastSAM-s.pt") # "FastSAM-s.pt" / "FastSAM-x.pt"

prev_frame = None
result, result_img = None, None

##


def process_frame_thread():

    global result, result_img
    previous_frame = None

    while True:
        with lock: # Lock ensures synchronization with frame capture

            if cap.isOpened():

                # Read from camera
                suc, frame = cap.read()
                if not suc: continue

                # Convert frame to grayscale for simpler comparison
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                frame_diff = np.average(np.abs(gray_frame.astype(int) - previous_frame.astype(int))) if previous_frame is not None else 999_999_999
                if frame_diff > frame_diff_epsilon:

                    result = sam(
                        source=frame,
                        device='cuda',
                        retina_masks=True,
                        imgsz=1024,
                        conf=0.4,
                        iou=0.9,
                    )

                    sam_prompt = FastSAMPrompt(frame, result, device='cuda')
                    annotation = sam_prompt.text_prompt(text=track_target) if track_target else sam_prompt.everything_prompt()

                    result_img = sam_prompt.plot_to_result(annotation)

                else: print('no need to recalc :)', 'frame_diff:', frame_diff)

                previous_frame = gray_frame

        time.sleep(1/fps)


# Start daemon thread
thread = threading.Thread(target=process_frame_thread, daemon=True)
thread.start()


##




# Main loop to show result
# Should be intergrated into ROS later
while True:
    if cap.isOpened():
        with lock: # Lock ensures synchronization with the processing thread
            suc, frame = cap.read()
            if not suc: continue

            # Display the current frame
            cv2.imshow('Camera Frame', frame)
            # Display the processed result
            if result_img is not None: cv2.imshow('Processed Result', result_img)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
