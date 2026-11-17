"""
Loads a published model of this repository for the benchmark adapters.

The inference code and the settings of the model are those in hf_space/, the files
that are published on the Hugging Face Hub. The checkpoint is taken from weights/ if
it is there and downloaded from the Hub otherwise.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HUB_REPO = "maxwoe/image-rotation-angle-estimation"


def load(checkpoint_filename=None):
    """Returns (predict function, number of parameters) for a checkpoint named in hf_space/config.json."""
    import torch

    spec = importlib.util.spec_from_file_location("published_model_cgd", ROOT / "hf_space" / "model_cgd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = json.loads((ROOT / "hf_space" / "config.json").read_text(encoding="utf-8"))
    entries = config["models"]
    info = entries[config["default_model"]] if checkpoint_filename is None else \
        next(e for e in entries.values() if e["filename"] == checkpoint_filename)

    checkpoint = ROOT / "weights" / info["filename"]
    if not checkpoint.exists():
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(repo_id=HUB_REPO, filename=info["filename"])

    model = module.CGDAngleEstimation.try_load(checkpoint_path=str(checkpoint), image_size=info["input_size"])
    model.apply_config(info)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model.predict_angle, sum(p.numel() for p in model.parameters())
