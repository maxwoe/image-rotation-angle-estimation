import gradio as gr
import torch
from PIL import Image
import os
from pathlib import Path
from loguru import logger
import numpy as np
from torchvision import transforms
import glob

# Import model classes
from model_unit_vector import UnitVectorAngleEstimation
from model_direct_angle import DirectAngleEstimation
from model_classification import ClassificationAngleEstimation
from model_cgd import CGDAngleEstimation
from model_psc import PSCAngleEstimation
from architectures import get_architecture_names, get_default_input_size

# Constants

# Initialize global model variable
model = None

# State object to manage model state
class ModelState:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.current_approach = None
        self.current_architecture = None
        self.current_checkpoint = None
        self.current_image_size = None
        self.current_device = None

state = ModelState()

# Available approaches and architectures
APPROACHES = {
    "unit_vector": UnitVectorAngleEstimation,
    "direct_angle": DirectAngleEstimation,
    "classification": ClassificationAngleEstimation,
    "cgd": CGDAngleEstimation,
    "psc": PSCAngleEstimation,
}

ARCHITECTURES = get_architecture_names()

def format_checkpoint_name(checkpoint_path):
    """Format checkpoint path for display, removing base paths to save space"""
    if not checkpoint_path:
        return checkpoint_path
    
    # Remove common base paths
    path = checkpoint_path
    base_paths_to_remove = [
        "data/saved_models/",
        "weights/",
        "./data/saved_models/",
        "./weights/"
    ]
    
    for base_path in base_paths_to_remove:
        if path.startswith(base_path):
            path = path[len(base_path):]
            break
    
    return path

def get_formatted_checkpoint_choices():
    """Get checkpoint choices with formatted display names"""
    checkpoints = find_all_checkpoints()
    # Return list of tuples: (display_name, actual_path)
    formatted_choices = []
    for ckpt in checkpoints:
        display_name = format_checkpoint_name(ckpt)
        formatted_choices.append((display_name, ckpt))
    return formatted_choices

def get_device():
    """Get the best available device"""
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def update_image_size(architecture):
    """Update image size based on selected architecture"""
    default_size = get_default_input_size(architecture)
    return gr.update(value=default_size)

def auto_load_model():
    """Automatically load the first available model"""
    global state
    if state.is_loaded:
        return gr.Info("Model already loaded successfully")
    
    try:
        checkpoints = find_all_checkpoints()
        if not checkpoints:
            return gr.Info("No checkpoint files found for auto-loading")
        
        device = get_device()
        result = load_selected_model(
            approach="cgd",
            architecture="convnextv2_base.fcmae_ft_in22k_in1k", 
            checkpoint_path=checkpoints[0],
            image_size=None
        )
        return result
    except Exception as e:
        return gr.Error(f"Failed to auto-load model: {str(e)}")

def get_model_path():
    """Get available model paths for dropdown"""
    checkpoints = find_all_checkpoints()
    if checkpoints:
        formatted_choices = get_formatted_checkpoint_choices()
        return gr.update(choices=formatted_choices, value=formatted_choices[0] if formatted_choices else None)
    return gr.update(choices=[], value=None)

def find_all_checkpoints():
    """Find all available checkpoint files"""
    checkpoint_patterns = [
        "weights/*.ckpt",
        "data/saved_models/**/*.ckpt"
    ]
    
    all_files = []
    for pattern in checkpoint_patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    # Sort by modification time (newest first)
    all_files.sort(key=os.path.getmtime, reverse=True)
    
    return all_files

def load_selected_model(approach, architecture, checkpoint_path, image_size=None):
    """Load model with selected parameters"""
    global state
    
    # Auto-detect device
    device = get_device()
    
    # Use provided image size or default for architecture
    final_image_size = image_size if image_size and image_size > 0 else get_default_input_size(architecture)
    
    # Check if model is already loaded with same parameters
    if (state.is_loaded and 
        state.current_approach == approach and 
        state.current_architecture == architecture and 
        state.current_checkpoint == checkpoint_path and 
        state.current_image_size == final_image_size and 
        state.current_device == device):
        return gr.Info("Model already loaded successfully")
    
    try:
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            # Create new model without checkpoint
            model_class = APPROACHES[approach]
            # Provide required parameters for all model types
            model_kwargs = {
                "model_name": architecture,
                "image_size": final_image_size,
                "batch_size": 16,  # Default batch size
                "train_dir": "dummy_path"  # Dummy path for inference-only
            }
            new_model = model_class(**model_kwargs)
            new_model.eval()
            
            # Update state
            state.model = new_model
            state.is_loaded = True
            state.current_approach = approach
            state.current_architecture = architecture
            state.current_checkpoint = "New model (not trained)"
            state.current_image_size = final_image_size
            state.current_device = device
            
            logger.warning(f"Created new {approach} model with {architecture} (not trained)")
            return gr.Info(f"Created new {approach} model with {architecture}. Model is not trained!")
        
        # Load from checkpoint
        model_class = APPROACHES[approach]
        
        # Use try_load if available, otherwise use load_from_checkpoint
        if hasattr(model_class, 'try_load'):
            try:
                new_model = model_class.try_load(checkpoint_path=checkpoint_path, image_size=final_image_size)
            except Exception as e:
                logger.warning(f"try_load failed: {e}. Attempting direct checkpoint loading...")
                # Fallback to direct checkpoint loading with required parameters
                load_kwargs = {
                    "image_size": final_image_size,
                    "batch_size": 16,
                    "train_dir": "dummy_path"
                }
                new_model = model_class.load_from_checkpoint(checkpoint_path, **load_kwargs)
        else:
            # Direct checkpoint loading with all required parameters
            load_kwargs = {
                "image_size": final_image_size,
                "batch_size": 16,
                "train_dir": "dummy_path"
            }
            new_model = model_class.load_from_checkpoint(checkpoint_path, **load_kwargs)
        
        new_model.eval()
        
        # Move to device if CUDA is available
        if torch.cuda.is_available() and device.startswith("cuda"):
            new_model = new_model.to(device)
        
        # Update state
        state.model = new_model
        state.is_loaded = True
        state.current_approach = approach
        state.current_architecture = architecture
        state.current_checkpoint = checkpoint_path
        state.current_image_size = final_image_size
        state.current_device = device
        
        logger.info(f"Loaded {approach} model from {checkpoint_path}")
        checkpoint_display_name = format_checkpoint_name(checkpoint_path)
        return gr.Info(f"Successfully loaded {approach} model ({checkpoint_display_name}) on {device}")
        
    except Exception as e:
        state.is_loaded = False
        state.model = None
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(error_msg)
        return gr.Error(error_msg)

def get_current_model():
    """Get the currently loaded model"""
    global state
    if not state.is_loaded or state.model is None:
        raise Exception("No model loaded. Please select and load a model first.")
    return state.model

def predict_angle(model, image_path):
    """Predict the orientation angle of an image using the model's built-in predict_angle method"""
    # Use the model's own predict_angle method which handles all the approach-specific logic
    if hasattr(model, 'predict_angle'):
        return model.predict_angle(image_path)
    else:
        # Fallback to manual prediction if predict_angle method is not available
        raise NotImplementedError(f"Model {type(model).__name__} does not have a predict_angle method")

def predict_and_correct_orientation(input_image):
    """
    Predict the orientation angle and return both original and corrected images
    """
    try:
        # Get current model
        current_model = get_current_model()
        
        if input_image is None:
            return None, "Please upload an image first."
        
        # Save temporary input image
        temp_input_path = "temp_input.jpg"
        input_image.save(temp_input_path)
        
        # Predict angle
        predicted_angle = predict_angle(current_model, temp_input_path)
        
        # Correct orientation (negative angle to counter-rotate)
        corrected_image = input_image.rotate(-predicted_angle, expand=True, fillcolor=(255, 255, 255))
        
        # Clean up temporary file
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        
        result_message = f"Predicted rotation angle: {predicted_angle:.2f}°\nModel: {state.current_approach} ({state.current_architecture})\nImage size: {state.current_image_size}x{state.current_image_size}"
        
        logger.info(f"Successfully processed image. Predicted angle: {predicted_angle:.2f}°")
        
        return corrected_image, result_message
        
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        logger.error(error_message)
        return None, error_message

def batch_process_images(input_files):
    """
    Process multiple images and return a gallery of corrected images
    """
    try:
        if not input_files:
            return None, "Please upload at least one image."
        
        current_model = get_current_model()
        corrected_images = []
        results = []
        
        for i, input_file in enumerate(input_files):
            try:
                # Process each image
                input_image = Image.open(input_file.name).convert('RGB')
                predicted_angle = predict_angle(current_model, input_file.name)
                corrected_image = input_image.rotate(-predicted_angle, expand=True, fillcolor=(255, 255, 255))
                
                corrected_images.append(corrected_image)
                results.append(f"Image {i+1}: {predicted_angle:.2f}° rotation")
                
            except Exception as e:
                results.append(f"Image {i+1}: Error - {str(e)}")
        
        result_message = f"Batch Processing Results ({state.current_approach}):\n" + "\n".join(results)
        logger.info(f"Batch processed {len(input_files)} images")
        
        return corrected_images, result_message
        
    except Exception as e:
        error_message = f"Error in batch processing: {str(e)}"
        logger.error(error_message)
        return None, error_message


if __name__ == "__main__":
    logger.info("Starting Image Rotation Amgle Estimation Gradio App")
    
    # Create interface
    app = gr.Blocks()
    with app:
        # app.load(auto_load_model)
        
        gr.HTML("<h1>Image Rotation Amgle Estimation</h1>")
        
        # Model Selection Section
        with gr.Accordion(label="Model Settings", open=False):
            get_model_path_btn = gr.Button("Get Models")
            checkpoint_dropdown = gr.Dropdown(
                label="Checkpoint Path",
                choices=get_formatted_checkpoint_choices(),
                value=None
            )
            
            with gr.Row():
                approach_dropdown = gr.Dropdown(
                    choices=list(APPROACHES.keys()),
                    value="cgd",
                    label="Approach",
                    info="Select the model approach"
                )
                architecture_dropdown = gr.Dropdown(
                    choices=ARCHITECTURES,
                    value="convnextv2_base.fcmae_ft_in1k",
                    label="Architecture", 
                    info="Select the model architecture"
                )
                image_size_input = gr.Number(
                    value=get_default_input_size("convnextv2_base.fcmae_ft_in1k"),
                    label="Image Size",
                    info="Auto-filled based on architecture",
                    precision=0,
                    minimum=32,
                    maximum=1024
                )
            
            load_model_btn = gr.Button("Load Model", variant="primary")
            
            get_model_path_btn.click(get_model_path, outputs=[checkpoint_dropdown])
            architecture_dropdown.change(
                update_image_size,
                inputs=[architecture_dropdown],
                outputs=[image_size_input]
            )
            load_model_btn.click(
                load_selected_model,
                inputs=[approach_dropdown, architecture_dropdown, checkpoint_dropdown, image_size_input],
                outputs=[]
            )
        
        with gr.Tabs():
            # Single Image Processing Tab
            with gr.Tab("Single Image"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            type="pil", 
                            label="Upload Image",
                            height=400
                        )
                        process_btn = gr.Button(
                            "Correct Orientation", 
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        output_image = gr.Image(
                            label="Corrected Image",
                            height=400
                        )
                        result_text = gr.Textbox(
                            label="Results",
                            lines=3,
                            max_lines=5
                        )
                
                process_btn.click(
                    fn=predict_and_correct_orientation,
                    inputs=[input_image],
                    outputs=[output_image, result_text]
                )
            
            # Batch Processing Tab
            with gr.Tab("Batch Processing"):
                with gr.Column():
                    batch_input = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload Multiple Images"
                    )
                    batch_process_btn = gr.Button(
                        "Process All Images", 
                        variant="primary",
                        size="lg"
                    )
                    
                    batch_gallery = gr.Gallery(
                        label="Corrected Images",
                        show_label=True,
                        elem_id="gallery",
                        columns=3,
                        rows=2,
                        height="auto"
                    )
                    
                    batch_results = gr.Textbox(
                        label="Batch Results",
                        lines=10,
                        max_lines=20
                    )
                
                batch_process_btn.click(
                    fn=batch_process_images,
                    inputs=[batch_input],
                    outputs=[batch_gallery, batch_results]
                )
            
    
    # Launch with public sharing disabled by default for security
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Use different port to avoid conflicts
        share=False,  # Set to True if you want to create a public link
        show_error=True
    )