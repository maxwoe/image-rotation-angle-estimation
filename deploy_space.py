"""Deploy the Gradio demo to HuggingFace Spaces.

Uploads the inference-only code from hf_space/ to the Space in one commit. The Space is
updated in place: its history, settings and URL are kept, and files that are no longer
part of the demo are removed from it.

Run this script on a machine where `huggingface-cli login` has been completed.

Usage:
    python deploy_space.py                      # upload
    python deploy_space.py --message "..."      # upload with a commit message of your own
"""

import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

REPO_ID = os.environ.get("HF_MODEL_REPO", "maxwoe/image-rotation-angle-estimation")

# Inference-only files from hf_space/ directory
HF_SPACE_DIR = "hf_space"
SPACE_FILES = [
    "app.py",
    "model_cgd.py",
    "architectures.py",
    "rotation_utils.py",
    "requirements.txt",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--message", default="Deploy inference-only Gradio demo", help="commit message on the Hub")
    args = parser.parse_args()

    api = HfApi()
    # The Hub treats the creation of a Gradio Space as a separate step that depends on the
    # plan of the account, so it is only requested when the Space does not exist yet.
    if not api.repo_exists(repo_id=REPO_ID, repo_type="space"):
        api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="gradio")

    staging_dir = tempfile.mkdtemp(prefix="hf_space_")
    print(f"Staging files in: {staging_dir}")
    try:
        # space_readme.md becomes the README.md of the Space (its configuration)
        shutil.copy2("space_readme.md", os.path.join(staging_dir, "README.md"))
        for filename in SPACE_FILES:
            shutil.copy2(os.path.join(HF_SPACE_DIR, filename), os.path.join(staging_dir, filename))
        shutil.copytree(os.path.join(HF_SPACE_DIR, "examples"), os.path.join(staging_dir, "examples"))

        print(f"Uploading to the Space {REPO_ID}...")
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type="space",
            folder_path=staging_dir,
            commit_message=args.message,
            delete_patterns="*",  # remove remote files that are not part of this upload
        )
        print(f"Done. Space URL: https://huggingface.co/spaces/{REPO_ID}")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
