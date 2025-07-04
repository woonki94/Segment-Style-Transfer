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


def read_video(video_path):
    cap_video = cv2.VideoCapture(video_path)
    if not cap_video.isOpened():
        print("Error: Could not open video file")
        exit()

    # Get video properties
    fps = int(cap_video.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

    return cap_video


def demo(opt):
    if torch.backends.mps.is_available():
        device = "mps"  # Use Apple's GPU
    elif torch.cuda.is_available():
        device = "cuda"  # For other GPUs (NVIDIA, etc.)
    else:
        device = "cpu"

    print(f"Using device: {device}")

    model = VOSNet(opt)
    model = load_model(model, opt)
    model.to(device)
    model.eval()

    # Load Transformer Networks
    print("Loading Transformer Networks")
    # load all the models in the folder
    nets = []
    for file in os.listdir("./fast-neural-style-pytorch/transforms"):
        if file.endswith(".pth"):
            print(file)
            net = transformer.TransformerNetwork()
            net.load_state_dict(
                torch.load(f"./fast-neural-style-pytorch/transforms/{file}", map_location=torch.device('cpu')))
            net = net.to(device)
            nets.append(net)

    total_nets = len(nets)
    current_net = nets[0]
    current_style = "Style 1: Lazy"
    background_net = nets[0]
    print("Done Loading Transformer Networks")

    cap_video1 = read_video("./videos/20250314_161953.mp4")
    cap_video2 = read_video("./videos/20250314_162110.mp4")
    cap_video3 = read_video("./videos/20250314_162236.mp4")
    cap_video4 = read_video("./videos/20250314_165623.mp4")
    cap_video5 = read_video("./videos/20250314_165717.mp4")
    background0 = read_video("./background_images/korean_screen_shot.mov")
    background1 = cv2.imread("./background_images/building.webp")

    # Initialize variables for optical flow
    prev_frame = None
    prev_gray = None
    fps = 0
    fps_alpha = 0.1  # Smoothing factor for FPS
    segment_mode = False
    speed_up = True
    video_number = 1
    video_combine = False
    background_video = 6
    style_number = 0
    background_style_number = 0
    no_change = True
    background_no_change = True
    with torch.no_grad():
        while True:
            start_time = time.perf_counter()

            if video_number == 1:
                ret, frame = cap_video1.read()
            elif video_number == 2:
                ret, frame = cap_video2.read()
            elif video_number == 3:
                ret, frame = cap_video3.read()
            elif video_number == 4:
                ret, frame = cap_video4.read()
            elif video_number == 5:
                ret, frame = cap_video5.read()

            if video_combine == True:
                if background_video == 6:
                    ret, background = background0.read()
                elif background_video == 7:
                    background = background1
                    ret = True
                if not ret:
                    print("[ERROR] Failed to grab frame1")
                    break

                background = cv2.resize(background, (640, 480))

            if not ret:
                print("[ERROR] Failed to grab frame1")
                break
            if speed_up == True:
                # read streamfrom video
                frame = cv2.resize(frame, (640, 480))

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

                # Generate image with style transfer
                if no_change == False:
                    content_tensor = style_transfer_utils.itot(overlay).to(device)
                    generated_tensor = current_net(content_tensor)
                    generated_image = style_transfer_utils.ttoi(generated_tensor.detach())
                else:
                    generated_image = overlay

                ### background change
                if video_combine == True:
                    if background_no_change == False:
                        content_tensor = style_transfer_utils.itot(background).to(device)
                        generated_tensor = background_net(content_tensor)
                        b_generated_image = style_transfer_utils.ttoi(generated_tensor.detach())
                    else:
                        b_generated_image = background

                output = F.interpolate(saliency, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

                sal = output.sigmoid().data.cpu().numpy().squeeze()
                threshold = 0.4
                sal_mask = (sal > threshold)
                background_mask = (sal < threshold)

                # Create the final composite image
                final_image = generated_image.copy()
                background_image = generated_image.copy()

                if video_combine == True:
                    background_image[background_mask] = b_generated_image[background_mask]
                else:
                    final_image[sal_mask] = frame[sal_mask]

                if video_combine == True:
                    display_image = background_image.copy()
                else:
                    display_image = final_image.copy()
                display_image = display_image / 255

                # Calculate and smooth FPS
                current_fps = 1.0 / (time.perf_counter() - start_time)
                fps = fps * (1 - fps_alpha) + current_fps * fps_alpha

                # Add FPS and style information
                # cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30),
                #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(display_image, current_style, (10, 70),
                #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show both the saliency map and the final result
                # vis = np.hstack([sal_map_colored, display_image])
                if segment_mode:
                    display_image = frame
                cv2.imshow('Result', display_image)

                prev_frame = frame
                prev_gray = gray

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    style_number += 1
                    if style_number == total_nets:
                        style_number = 0
                    if style_number == 0:
                        no_change = True
                    if style_number == 1:
                        no_change = False
                    current_net = nets[style_number]
                elif key == ord('v'):
                    background_style_number += 1
                    if background_style_number == total_nets:
                        background_style_number = 0
                    if background_style_number == 0:
                        background_no_change = True
                    if background_style_number == 1:
                        background_no_change = False
                    background_net = nets[background_style_number]
                elif key == ord('s'):
                    segment_mode = not segment_mode
                    print(f"Segment mode: {segment_mode}")
                elif key == ord('m'):
                    video_number += 1
                    if video_number == 6:
                        video_number = 1
                elif key == ord('n'):
                    background_video += 1
                    if background_video == 13:
                        background_video = 6
                elif key == ord('b'):
                    video_combine = not video_combine
                    print(f"Video combine: {video_combine}")
            speed_up = not speed_up
    # Cleanup
    #cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo(opt=opt)