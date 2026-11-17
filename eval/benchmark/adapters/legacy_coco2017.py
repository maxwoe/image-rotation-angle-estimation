"""
Adapter for the legacy COCO 2017 checkpoint (CGD + MambaOut Base, conventional protocol).

This model was trained on views that were never saved after rotation and reads the
rotated JPEG block grid of the source photos. It is in the benchmark as the example of
a model that the check for processing traces is there to detect.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _repository_model  # noqa: E402

INFO = {
    "name": "Legacy: CGD + MambaOut Base (COCO 2017, conventional protocol)",
    "parameters": None,  # counted in load()
    "training_data": "COCO 2017 train without the 915 test photos (117,372 images), 10% of them held out for validation; "
                     "training views not re-encoded after rotation",
    "input": "224 x 224 pixels: shorter side resized to 224, central square",
    "source": "https://huggingface.co/maxwoe/image-rotation-angle-estimation",
}


def load():
    predict, INFO["parameters"] = _repository_model.load("cgd_mambaout_base_coco2017.ckpt")
    return predict
