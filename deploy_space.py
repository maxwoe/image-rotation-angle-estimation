"""Deploy the Gradio demo to HuggingFace Spaces.

Run this script on the machine where huggingface-cli login has been completed.

Usage:
    python deploy_space.py
"""

import os
import shutil
import tempfile
from huggingface_hub import HfApi, upload_folder

REPO_ID = os.environ.get("HF_MODEL_REPO", "maxwoe/image-rotation-angle-estimation")

# Files to upload to the Space (only what app_hf.py transitively imports)
# app_hf.py -> model_cgd.py -> data_loader.py -> rotation_utils.py
#                            -> metrics.py
#           -> architectures.py
SPACE_FILES = [
    "model_cgd.py",
    "data_loader.py",
    "rotation_utils.py",
    "metrics.py",
    "architectures.py",
    "requirements.txt",
]


def main():
    api = HfApi()

    # Step 1: Create the Space (no-op if it already exists)
    print(f"Creating Space: {REPO_ID}")
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )
        print("Space created (or already exists).")
    except Exception as e:
        print(f"Note: {e}")

    # Step 2: Stage files in a temp directory
    staging_dir = tempfile.mkdtemp(prefix="hf_space_")
    print(f"Staging files in: {staging_dir}")

    try:
        # Copy app_hf.py as app.py (Space expects app.py)
        shutil.copy2("app_hf.py", os.path.join(staging_dir, "app.py"))

        # Copy space_readme.md as README.md (Space config)
        shutil.copy2("space_readme.md", os.path.join(staging_dir, "README.md"))

        # Copy all other files
        for filename in SPACE_FILES:
            if os.path.exists(filename):
                shutil.copy2(filename, os.path.join(staging_dir, filename))
            else:
                print(f"Warning: {filename} not found, skipping")

        # Step 3: Upload everything
        print(f"Uploading to {REPO_ID}...")
        upload_folder(
            repo_id=REPO_ID,
            repo_type="space",
            folder_path=staging_dir,
            commit_message="Deploy Gradio demo",
        )
        print(f"Done! Space URL: https://huggingface.co/spaces/{REPO_ID}")

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
