import os
import _init_paths
from config import Parameters

opt = Parameters().parse()

from style_utils import style_transfer_utils, transformer
import time
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from lib.builder import VOSNet
from utils.init import load_model


def preprocess_frame(frame, img_size):
    """Preprocess frame for model input"""
    frame = cv2.resize(frame, (img_size, img_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    frame = torch.from_numpy(frame.transpose(2, 0, 1)).float().unsqueeze(0)
    return frame


def postprocess_output(output, orig_h, orig_w):
    """Convert model output to displayable format"""
    output = F.interpolate(output, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    sal = output.sigmoid().data.cpu().numpy().squeeze()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)  # Normalize
    sal = (sal * 255).astype(np.uint8)
    sal_colored = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
    return sal, sal_colored


def demo(opt):
    # Select best available device
    if torch.backends.mps.is_available():
        device = "mps"  # Apple GPU
    elif torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load segmentation model
    model = VOSNet(opt)
    model = load_model(model, opt)
    model.to(device)
    model.eval()

    # Load Transformer Networks (Style Transfer Models)
    print("Loading Transformer Networks")
    net1 = transformer.TransformerNetwork()
    net1.load_state_dict(torch.load("../fast-neural-style-pytorch/transforms/lazy.pth", map_location=torch.device(device)))
    net1.to(device)

    net2 = transformer.TransformerNetwork()
    net2.load_state_dict(torch.load("../fast-neural-style-pytorch/transforms/wave.pth", map_location=torch.device(device)))
    net2.to(device)

    current_net = net1
    current_style = "Style 1: Lazy"
    print("Done Loading Transformer Networks")

    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Starting real-time inference...")

    prev_frame = None
    prev_gray = None
    fps = 0
    fps_alpha = 0.1  # Smoothing factor for FPS

    with torch.no_grad():
        while True:
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            orig_h, orig_w = frame.shape[:2]

            # Preprocess input
            input_frame = preprocess_frame(frame, opt.img_size).to(device)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute optical flow
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow = np.concatenate([flow, np.zeros_like(flow[:, :, 0:1])], axis=2)
                flow = torch.from_numpy(flow.transpose(2, 0, 1)).float().unsqueeze(0)
                flow = F.interpolate(flow, size=(opt.img_size, opt.img_size), mode='bilinear', align_corners=False)
                flow = flow.to(device)
            else:
                flow = torch.zeros_like(input_frame)

            # Prepare content for style transfer
            content_tensor = style_transfer_utils.itot(frame).to(device)

            saliency_future = torch.jit.fork(model, input_frame, flow)  # Saliency detection
            style_future = torch.jit.fork(current_net, content_tensor)  # Style transfer

            # Wait for results
            saliency = torch.jit.wait(saliency_future)
            generated_tensor = torch.jit.wait(style_future)

            # Post-process saliency
            sal_map, sal_map_colored = postprocess_output(saliency, orig_h, orig_w)

            # Convert generated image to displayable format
            generated_image = style_transfer_utils.ttoi(generated_tensor.detach())

            # Create binary mask from saliency map
            output = F.interpolate(saliency, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            sal = output.sigmoid().data.cpu().numpy().squeeze()
            threshold = 0.4  # Saliency threshold
            sal_mask = (sal > threshold)

            # Composite image: Keep original image where saliency is detected
            final_image = generated_image.copy()
            final_image[sal_mask] = frame[sal_mask]

            # Convert final image to display format
            display_image = final_image.copy() / 255.0

            # Calculate and smooth FPS
            current_fps = 1.0 / (time.perf_counter() - start_time)
            fps = fps * (1 - fps_alpha) + current_fps * fps_alpha

            # Display FPS & style info
            cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_image, current_style, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display results
            vis = np.hstack([sal_map_colored, display_image])
            cv2.imshow('Saliency Map | Result', vis)

            # Update previous frame
            prev_frame = frame
            prev_gray = gray

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('c'):  # Change style
                if current_net is net1:
                    current_net = net2
                    current_style = "Style 2: Wave"
                else:
                    current_net = net1
                    current_style = "Style 1: Lazy"
                print(f"Switched to {current_style}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo(opt=opt)
