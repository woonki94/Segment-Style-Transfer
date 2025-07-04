import os
import _init_paths
from config import Parameters

opt = Parameters().parse()

from style_utils import style_transfer_utils, transformer
import time
import torch
from torch.utils import data
from lib.builder import VOSNet
import cv2
import numpy as np
from utils.init import load_model
import torch.nn.functional as F


def preprocess_frame(frame, img_size):
    """Preprocess frame for model input"""

    frame = cv2.resize(frame, (img_size, img_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    frame = torch.from_numpy(frame.transpose(2, 0, 1)).float().unsqueeze(0)
    return frame


def postprocess_output(output, orig_h, orig_w):
    """Convert model output to displayable format using same method as save_images()"""

    output = F.interpolate(output, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

    sal = output.sigmoid().data.cpu().numpy().squeeze()

    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

    sal = (sal * 255).astype(np.uint8)

    sal_colored = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
    return sal, sal_colored


def demo(opt):
    if torch.backends.mps.is_available():
        device = "mps"  # Use Apple's GPU
    elif torch.cuda.is_available():
        device = "cuda"  # For other GPUs (NVIDIA, etc.)
    else:
        device = "cpu"

    model = VOSNet(opt)
    model = load_model(model, opt)
    model.to(device)
    model.eval()

    print(f"Using device: {device}")

    # Load Transformer Networks
    print("Loading Transformer Networks")
    net1 = transformer.TransformerNetwork()

    net1.load_state_dict(torch.load("./fast-neural-style-pytorch/transforms/lazy.pth",
                                    map_location=torch.device('cpu')))
    net1 = net1.to(device)
    net2 = transformer.TransformerNetwork()
    net2.load_state_dict(torch.load("./fast-neural-style-pytorch/transforms/wave.pth",
                                    map_location=torch.device('cpu')))
    net2 = net2.to(device)

    current_net = net1
    current_style = "Style 1: Lazy"
    print("Done Loading Transformer Networks")

    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Starting real-time inference...")

    # Initialize variables for optical flow
    prev_frame = None
    prev_gray = None
    fps = 0
    fps_alpha = 0.1  # Smoothing factor for FPS
    segment_mode = False

    with torch.no_grad():
        while True:
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            orig_h, orig_w = frame.shape[:2]

            input_frame = preprocess_frame(frame, opt.img_size)
            input_frame = input_frame.to(device)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow = np.concatenate([flow, np.zeros_like(flow[:, :, 0:1])], axis=2)
                flow = torch.from_numpy(flow.transpose(2, 0, 1)).float().unsqueeze(0)
                flow = F.interpolate(flow, size=(opt.img_size, opt.img_size), mode='bilinear', align_corners=False)
                flow = flow.to(device)
            else:
                flow = torch.zeros_like(input_frame)

            # Forward pass
            saliency = model(input_frame, flow)

            sal_map, sal_map_colored = postprocess_output(saliency, orig_h, orig_w)

            overlay = frame.copy()
            # img = cv2.flip(overlay, 1)

            # Generate image with style transfer
            content_tensor = style_transfer_utils.itot(overlay).to(device)
            generated_tensor = current_net(content_tensor)
            generated_image = style_transfer_utils.ttoi(generated_tensor.detach())

            output = F.interpolate(saliency, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

            sal = output.sigmoid().data.cpu().numpy().squeeze()
            threshold = 0.4
            sal_mask = (sal > threshold)

            # Create the final composite image
            final_image = generated_image.copy()
            # Keep original image in salient regions (where sal_mask is True)
            final_image[sal_mask] = frame[sal_mask]

            # Convert to display format
            display_image = final_image.copy()
            display_image = display_image / 255

            # Calculate and smooth FPS
            current_fps = 1.0 / (time.perf_counter() - start_time)
            fps = fps * (1 - fps_alpha) + current_fps * fps_alpha

            # Add FPS and style information
            cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_image, current_style, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show both the saliency map and the final result
            vis = np.hstack([sal_map_colored, display_image])
            cv2.imshow('Saliency Map | Result', vis)

            prev_frame = frame
            prev_gray = gray

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Switch between networks
                if current_net is net1:
                    current_net = net2
                    current_style = "Style 2: Wave"
                else:
                    current_net = net1
                    current_style = "Style 1: Lazy"
                print(f"Switched to {current_style}")
            elif key == ord('s'):
                segment_mode = not segment_mode
                print(f"Segment mode: {segment_mode}")
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo(opt=opt)