import os
import cv2
import torch
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
import torchvision.transforms as transforms
from model import UNet

def post_process_prediction(pred_mask, original_image_np):
    """
    Processes a raw probability float mask and extracts structural bounding contours.
    Filters out micro-artifacts and overlays target boxes onto a numpy canvas image.
    """
    # Scale from probabilities [0, 1] to traditional image arrays [0, 255]
    mask_uint8 = (pred_mask * 255).astype(np.uint8)
    
    # Otsu thresholding automatically separates background/foreground
    _, thresh = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Extract outer contours 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an active working canvas
    output_img = original_image_np.copy()
    box_count = 0
    
    for cnt in contours:
        # Ignore noisy point clusters (less than 50 pixels in area size)
        if cv2.contourArea(cnt) > 50: 
            # Use native bounding coordinates extraction 
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Overlay a green contour line and a distinct red localization box
            cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 1)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            box_count += 1
            
    print(f"Post-processing complete. Detected {box_count} target regions.")
    return mask_uint8, thresh, output_img

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoint/bird_segmentation_v1.pth"
    test_image_path = "./test/Bild09.png"  # Adjust this path to any image in your test folder
    
    # Setup standard inference pipeline transforms
    transform_pipeline = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])
    
    # Initialize and load model configuration
    model = UNet()
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from system file: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("WARNING: Checkpoint path not found. Running inference with randomized weights.")
        
    model.to(device).eval()

    if not os.path.exists(test_image_path):
        print(f"Error: Target image file '{test_image_path}' not found. Please verify location path.")
    else:
        # Load and convert image to matching evaluation dims
        orig_image = Image.open(test_image_path).convert("RGB")
        tensor_img = transform_pipeline(orig_image).to(device).unsqueeze(0)
        
        # Forward pass extraction 
        with torch.no_grad():
            device_type = "cuda" if "cuda" in device else "cpu"
            with torch.amp.autocast(device_type=device_type):
                raw_output = model(tensor_img)
            pred_mask = torch.sigmoid(raw_output).cpu().squeeze().numpy()
        
        # Prepare matching numpy array layout for plotting [H, W, C]
        resized_image_np = np.moveaxis(transform_pipeline(orig_image).numpy(), 0, -1)
        
        # Run computer vision optimization functions
        raw_mask, binary_thresh, final_localization = post_process_prediction(pred_mask, resized_image_np)
        
        # Render clean diagnostic visualization plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(raw_mask, cmap='gray')
        axes[0].set_title("1. Raw Prediction Intensity Map")
        axes[0].axis('off')
        
        axes[1].imshow(binary_thresh, cmap='gray')
        axes[1].set_title("2. Otsu's Binary Threshold Mask")
        axes[1].axis('off')
        
        # Ensure values fall within expected bounding range for safe plotting
        display_img = np.clip(final_localization, 0, 1) if final_localization.max() <= 1.0 else final_localization
        axes[2].imshow(display_img)
        axes[2].set_title("3. Bounding Box & Contour Localization")
        axes[2].axis('off')
        
        plt.tight_layout()
        print("Displaying evaluation figure window layout stack...")
        #plt.show()
        plt.savefig("test_output_localization.png", dpi=300, bbox_inches='tight')
        plt.show()