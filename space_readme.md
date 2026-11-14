---
title: Image Rotation Angle Estimation
emoji: "\U0001F504"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.32.0"
app_file: app.py
python_version: "3.11"
pinned: false
---

Upload an image to predict and correct its rotation angle.
The default model uses the CGD (Circular Gaussian Distribution) head on a ConvNeXt V2 Base backbone, trained on COCO 2017 under a trace-free protocol, and reaches 3.77° MAE on COCO test views that are saved as JPEG after rotation.

[GitHub](https://github.com/maxwoe/image-rotation-angle-estimation) | [Paper](https://arxiv.org/abs/2603.25351) | [Pretrained Models](https://huggingface.co/maxwoe/image-rotation-angle-estimation)
