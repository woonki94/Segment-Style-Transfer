import os
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


def demo_video(opt, video_path, output_path=None):
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
    net1.load_state_dict(torch.load("../fast-neural-style-pytorch/transforms/lazy.pth",
                                    map_location=torch.device('cpu')))
    net1 = net1.to(device)
    net2 = transformer.TransformerNetwork()
    net2.load_state_dict(torch.load("../fast-neural-style-pytorch/transforms/wave.pth",
                                    map_location=torch.device('cpu')))
    net2 = net2.to(device)

    current_net = net1
    current_style = "Style 1: Lazy"
    print("Done Loading Transformer Networks")

    # Load video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w * 2, orig_h))

    print(f"[INFO] Processing video: {video_path}, FPS: {fps}, Resolution: {orig_w}x{orig_h}, Frames: {total_frames}")

    prev_frame = None
    prev_gray = None
    fps_alpha = 0.1  # Smoothing factor for FPS
    segment_mode = False

    with torch.no_grad():
        while True:
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video reached.")
                break

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
            final_image[~sal_mask] = frame[~sal_mask]

            display_image = final_image / 255

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
            #cv2.imshow('Saliency Map | Result', vis)

            if output_path:
                out.write((vis * 255).astype(np.uint8))  # Convert back to 8-bit format

            prev_frame = frame
            prev_gray = gray
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


def resize_video(video_path, target_fps=50):
    print("Resizing")
    output_path = "./resized_video.mp4"

    cap = cv2.VideoCapture(video_path)

    # Get original width, height, and FPS
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Original FPS: {original_fps}")

    # New target size
    target_width = 480
    target_height = 848

    # Video writer with new FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (target_width, target_height))

    frame_interval = original_fps // target_fps  # Skip frames to reduce FPS

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:  # Keep only every Nth frame
            frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            out.write(frame_resized)

        frame_count += 1

    cap.release()
    out.release()

    print(f"Resized video saved with {target_fps} FPS")
    return output_path


if __name__ == "__main__":
    video_path = "./43214321.mp4"  # Replace with your video path
    video_path = resize_video(video_path)
    output_path = "./output_video2.mp4"  # Set to None if you don't want to save output

    demo_video(opt=opt, video_path=video_path, output_path=output_path)