import cv2

from algorithms.fast_sam import fastSamRealTime


model_size = "small"
real_time_sam = fastSamRealTime(model_size=model_size)

# Initiate camera -> 0:embedded camera of laptop, 1,2,3...: external cam devs
cap = cv2.VideoCapture(0)

while cap.isOpened():

    suc, frame = cap.read()

    everything_results = real_time_sam.model(
        source=frame,
        device=real_time_sam.device,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )

    img = real_time_sam.wrapper_fastsam_prompt(frame, everything_results)

    cv2.imshow('frame', frame)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
