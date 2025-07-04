import os

from mrcnn.utils import resize_image
from streamlit import image

import _init_paths
from config import Parameters

opt = Parameters().parse()
from style_utils import style_transfer_utils, transformer
import time
import torch
from lib.builder import VOSNet
import cv2
import numpy as np
from utils.init import load_model
import torch.nn.functional as F

opt = Parameters().parse()




def resize_image(image_path,target_width=856, target_height=480):
    """
    Resizes an image to the target width and height and saves the resized image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
        target_width (int): Desired width of the output image.
        target_height (int): Desired height of the output image.
    """
    print(f"Resizing image: {image_path}")

    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot open image file: {image_path}")
        return None

    # Resize the image
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Save the resized image
    cv2.imwrite(output_path, resized_image)

    print(f"Resized image saved at: {output_path}")
    return output_path


def preprocess_image(image, img_size):
    """Preprocess image for model input"""
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    return image

def postprocess_output(output, orig_h, orig_w):
    """Convert model output to displayable format"""
    output = F.interpolate(output, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    sal = output.sigmoid().data.cpu().numpy().squeeze()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal = (sal * 255).astype(np.uint8)
    sal_colored = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
    return sal, sal_colored

def process_image(opt, image_path, flow_path=None, output_path=None):
    """Process a single image with a flow image instead of computed optical flow"""

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    model = VOSNet(opt)
    model = load_model(model, opt)
    model.to(device)
    model.eval()

    # Load Transformer Network for Style Transfer
    net1 = transformer.TransformerNetwork()
    net1.load_state_dict(torch.load("../fast-neural-style-pytorch/transforms/abstract.pth", map_location=device))
    net1 = net1.to(device)

    # Load Image
    image = cv2.imread(image_path)  # Load input image
    if image is None:
        print(f"[ERROR] Cannot open image file: {image_path}")
        return
    print(image.shape)
    orig_h, orig_w = image.shape[:2]
    input_image = preprocess_image(image, opt.img_size).to(device)

    # Load Flow Image as 3-Channel Image
    if flow_path:
        flow = cv2.imread(flow_path)  # Load as RGB (3-channel)
        if flow is None:
            print(f"[ERROR] Cannot open flow image: {flow_path}")
            return
        flow = cv2.resize(flow, (opt.img_size, opt.img_size))  # Resize
        flow = torch.from_numpy(flow.transpose(2, 0, 1)).float().unsqueeze(0)  # Convert to tensor
        flow = flow.to(device)  # Move to device
    else:
        flow = torch.zeros_like(input_image)  # Use zero flow (3 channels)

    # Forward Pass
    with torch.no_grad():
        saliency = model(input_image, flow)  # Pass input & flow

    # Post-process Output
    sal_map, sal_map_colored = postprocess_output(saliency, orig_h, orig_w)

    # Style Transfer
    content_tensor = style_transfer_utils.itot(image).to(device)
    generated_tensor = net1(content_tensor)
    generated_image = style_transfer_utils.ttoi(generated_tensor.detach())

    # Apply Saliency Mask
    output = F.interpolate(saliency, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    sal = output.sigmoid().data.cpu().numpy().squeeze()
    threshold = 0.4
    sal_mask = (sal > threshold)

    final_image = generated_image.copy()
    final_image[sal_mask] = image[sal_mask]

    # Save Results
    if output_path:
        cv2.imwrite(output_path, final_image)
        print(f"Processed image saved at: {output_path}")

    return final_image

if __name__ == "__main__":
    image_path =  "./horsejump.jpg"# Replace with your image path
    flow_path =   "./horse_flow.jpg" # Provide flow image path or set to None
    output_path = "./horse_abstract.jpg"  # Set to None if you don't want to save output

    image_path = resize_image(image_path)
    #flow_path = resize_image(flow_path)


    process_image(opt=opt, image_path=image_path, flow_path=flow_path, output_path=output_path)
