"""Deploy the Gradio demo to HuggingFace Spaces.

Uploads inference-only code from hf_space/ to the Space.
Deletes and recreates the Space to ensure clean git history.

Run this script on the machine where huggingface-cli login has been completed.

Usage:
    python deploy_space.py
"""

import os
import shutil
import tempfile
from huggingface_hub import HfApi, upload_folder

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
    api = HfApi()

    # Step 1: Delete existing Space to clear git history
    print(f"Deleting existing Space: {REPO_ID}")
    try:
        api.delete_repo(repo_id=REPO_ID, repo_type="space")
        print("Existing Space deleted.")
    except Exception as e:
        print(f"Note (delete): {e}")

    # Step 2: Create fresh Space
    print(f"Creating fresh Space: {REPO_ID}")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="gradio",
    )
    print("Space created.")

    # Step 3: Stage files in a temp directory
    staging_dir = tempfile.mkdtemp(prefix="hf_space_")
    print(f"Staging files in: {staging_dir}")

    try:
        # Copy space_readme.md as README.md (Space config)
        shutil.copy2("space_readme.md", os.path.join(staging_dir, "README.md"))

        # Copy inference-only files from hf_space/
        for filename in SPACE_FILES:
            src = os.path.join(HF_SPACE_DIR, filename)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(staging_dir, filename))
            else:
                print(f"Warning: {src} not found, skipping")

        # Copy example images
        examples_src = os.path.join(HF_SPACE_DIR, "examples")
        if os.path.isdir(examples_src):
            shutil.copytree(examples_src, os.path.join(staging_dir, "examples"))
            print(f"Copied {len(os.listdir(examples_src))} example images")

        # Step 4: Upload everything
        print(f"Uploading to {REPO_ID}...")
        upload_folder(
            repo_id=REPO_ID,
            repo_type="space",
            folder_path=staging_dir,
            commit_message="Deploy inference-only Gradio demo",
        )
        print(f"Done! Space URL: https://huggingface.co/spaces/{REPO_ID}")

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
