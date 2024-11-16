from fastsam import FastSAM, FastSAMPrompt
import torch
import numpy as np


# Two models are available now : FastSAM-s.pt and FastSAM-x.pt
# FastSAM-s is faster but less accurate
fast_sam_model = {
    "small": "FastSAM-s.pt",  # faster but less accurate
    "large": "FastSAM-x.pt"   # slower but more accurate
}


class fastSamRealTime():
    def __init__(self, model_size: str, object_to_track: str = None) -> None:
        if model_size not in fast_sam_model.keys():
            print(f"Entered model size {model_size} could not be found")
            print("Small model size is assigned instead!")
            model_size = "small"
        self.model = FastSAM(fast_sam_model[model_size])
        self.object_to_track = object_to_track
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"\n Used device for {self.__class__.__name__} = {self.device}")

    def wrapper_fastsam_prompt(self, frame, everything_results) -> np.ndarray:
        # TODO: This function only handles everything and textprompts.
        # TODO cont'd:  should be able to handle box and point prompts aswell.
        """Segment the image based on user request."""
        prompt_process = FastSAMPrompt(
            frame, everything_results, device=self.device
        )
        if self.object_to_track is None:
            ann = prompt_process.everything_prompt()
        else:
            ann = prompt_process.text_prompt(self.object_to_track)
        return prompt_process.plot_to_result(frame, annotations=ann)


def main():
    model_size = "large"
    object_to_track = "airplanes"
    model = fastSamRealTime(model_size, object_to_track)
    print(f"used device is {model.device}")


if __name__ == "__main__":
    main()
