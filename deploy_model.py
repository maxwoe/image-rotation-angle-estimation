"""Publish the model repository on the HuggingFace Hub.

Uploads, in one commit, the model card with its figure, config.json and the inference-only code from
hf_space/, together with every checkpoint named in config.json that is present in
weights/. Files on the Hub that are not part of the upload stay as they are, so
checkpoints that exist only on the Hub remain available.

The commit is refused if config.json names a checkpoint that is neither in weights/ nor
already on the Hub, so the published configuration never points to a missing file.

Run this script on a machine where `huggingface-cli login` has been completed.

Usage:
    python deploy_model.py            # upload
    python deploy_model.py --dry-run  # list what would be uploaded
"""

import argparse
import json
import os

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = os.environ.get("HF_MODEL_REPO", "maxwoe/image-rotation-angle-estimation")
HF_SPACE_DIR = "hf_space"
WEIGHTS_DIR = "weights"

# local file in hf_space/ -> path in the model repository
REPO_FILES = {
    "model_card.md": "README.md",
    "config.json": "config.json",
    "example.jpg": "example.jpg",
    "model_cgd.py": "model_cgd.py",
    "architectures.py": "architectures.py",
    "rotation_utils.py": "rotation_utils.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="list the files without uploading")
    args = parser.parse_args()

    api = HfApi()
    on_hub = set(api.list_repo_files(repo_id=REPO_ID, repo_type="model"))

    with open(os.path.join(HF_SPACE_DIR, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    if config["default_model"] not in config["models"]:
        raise SystemExit(f"default_model '{config['default_model']}' is not one of the models in config.json")

    operations = []
    for name, info in config["models"].items():
        local = os.path.join(WEIGHTS_DIR, info["filename"])
        if os.path.isfile(local):
            # content already on the Hub is recognised by its hash and not sent again
            size_mb = os.path.getsize(local) / 1e6
            print(f"checkpoint  {info['filename']}  ({size_mb:.0f} MB, from {WEIGHTS_DIR}/)")
            operations.append(CommitOperationAdd(path_in_repo=info["filename"], path_or_fileobj=local))
        elif info["filename"] in on_hub:
            print(f"checkpoint  {info['filename']}  (already on the Hub, kept)")
        else:
            raise SystemExit(f"config.json names '{info['filename']}' ({name}), "
                             f"which is neither in {WEIGHTS_DIR}/ nor on the Hub")

    for local_name, repo_path in REPO_FILES.items():
        local = os.path.join(HF_SPACE_DIR, local_name)
        print(f"file        {local} -> {repo_path}")
        operations.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local))

    if args.dry_run:
        print(f"dry run: {len(operations)} files would be committed to {REPO_ID}")
        return

    print(f"Uploading to the model repository {REPO_ID}...")
    api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=operations,
        commit_message="Publish model card, configuration, inference code and checkpoints",
    )
    print(f"Done. Model URL: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
