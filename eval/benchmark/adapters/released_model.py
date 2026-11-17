"""
Adapter for the released model of this repository (the default model on the Hugging Face Hub).

It is also the template for other models: describe the model in INFO, load it once in
load(), and return a function that takes a PIL image and returns the counter-clockwise
rotation of its content in degrees.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _repository_model  # noqa: E402

INFO = {
    "name": "CGD + ConvNeXt V2 Base (COCO 2017, trace-free protocol)",
    "parameters": None,  # counted in load()
    "training_data": "COCO 2017 train without the 915 test photos (117,372 images), 10% of them held out for validation; "
                     "every training view saved as JPEG after rotation",
    "input": "224 x 224 pixels: shorter side resized to 224, central square",
    "source": "https://huggingface.co/maxwoe/image-rotation-angle-estimation",
}


def load():
    predict, INFO["parameters"] = _repository_model.load()
    return predict
