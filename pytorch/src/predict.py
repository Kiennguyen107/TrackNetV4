"""
Script for predicting trajectories on video using a trained model.
It processes the video, performs inference using the loaded model, and outputs the predictions
as annotated video and CSV files with trajectory points.

Usage:
    python script.py --video_path <path_to_video> --model_weights <path_to_model_weights> --output_dir <output_directory> [--queue_length <queue_length>]
"""

import os
import sys
import cv2
import numpy as np
import time
import queue
import argparse
import torch
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from collections import deque
from models.TrackNetV2_pt import TrackNetV2 as TrackNetV2_pt
from models.TrackNetV4_pt import TrackNetV4 as TrackNetV4_pt
from models.TrackNetV5 import TrackNetV5
from util import get_model
from constants import HEIGHT, WIDTH

# Constants
BATCH_SIZE = 1
INPUT_HEIGHT = 288
INPUT_WIDTH = 512

def run_model_inference(model, frames, device):
    """
    Pre-processes the frames and runs inference on the model.
    
    Args:
        model: Loaded model for inference.
        frames (list): List of frames to run inference on.
        device: The device to run inference on.
    
    Returns:
        Tuple of predictions: (ball_predictions, player_predictions)
        inference_time (float): Time taken for model inference.
    """
    
    # Preprocess the frames for model input
    transform = ToTensor()
    resize = Resize((INPUT_HEIGHT, INPUT_WIDTH))
    
    input_batch = []
    for frame in frames:
        img = Image.fromarray(frame[..., ::-1])
        img = resize(img)
        img_tensor = transform(img)
        input_batch.append(img_tensor)

    # Prepare input for model prediction
    input_tensor = torch.cat(input_batch, dim=0).unsqueeze(0).to(device)

    # Perform prediction
    inference_start_time = time.time()
    with torch.no_grad():
        if isinstance(model, TrackNetV5):
            ball_preds, player_preds, _ = model(input_tensor)
            predictions = (ball_preds.cpu(), player_preds.cpu())
        elif isinstance(model, TrackNetV4_pt):
            ball_preds, _ = model(input_tensor)
            predictions = (ball_preds.cpu(), None)
        else:
            ball_preds = model(input_tensor)
            predictions = (ball_preds.cpu(), None)
    inference_end_time = time.time()

    inference_time = inference_end_time - inference_start_time
    return predictions, inference_time

def post_process_predictions(ball_predictions, player_predictions, frame1, frame2, frame3, frame_count, video_writer, ball_csv_path, player_csv_path, width_ratio, height_ratio, predicted_points_queue):
    """
    Post-processes the predictions, annotates the video, and saves results.
    """
    
    def process_heatmap(predictions, frame, current_frame_count, csv_path, width_ratio, height_ratio, is_player=False):
        binary_predictions = (predictions > 0.5).float()
        binary_heatmaps = (binary_predictions[0] * 255).byte().numpy()
        
        annotated_frame = frame.copy()

        for i in range(binary_heatmaps.shape[0]):
            heatmap = binary_heatmaps[i]
            if np.amax(heatmap) <= 0:
                if not is_player:
                    with open(csv_path, 'a') as f:
                        f.write(f"{current_frame_count + i},0,0,0\n")
            else:
                contours, _ = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if is_player:
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        x1 = int(width_ratio * x)
                        y1 = int(height_ratio * y)
                        x2 = int(width_ratio * (x + w))
                        y2 = int(height_ratio * (y + h))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        with open(csv_path, 'a') as f:
                            f.write(f"{current_frame_count + i},{x1},{y1},{x2},{y2}\n")
                else: # Ball
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    predicted_x_center = int(width_ratio * (x + w / 2))
                    predicted_y_center = int(height_ratio * (y + h / 2))

                    predicted_points_queue.appendleft((predicted_x_center, predicted_y_center))
                    
                    for point in predicted_points_queue:
                        if point is not None:
                            cv2.circle(annotated_frame, point, 5, (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (predicted_x_center, predicted_y_center), 5, (0, 0, 255), -1)

                    with open(csv_path, 'a') as f:
                        f.write(f"{current_frame_count + i},{predicted_x_center},{predicted_y_center},1\n")
        
        return annotated_frame

    frames = [frame1, frame2, frame3]
    
    # Process ball predictions
    ball_heatmaps = (ball_predictions > 0.5).float()
    ball_binary_heatmaps = (ball_heatmaps[0] * 255).byte().numpy()

    annotated_frames = [f.copy() for f in frames]

    for i, current_frame in enumerate(frames):
        # Ball processing
        if np.amax(ball_binary_heatmaps[i]) <= 0:
            with open(ball_csv_path, 'a') as csv_file:
                csv_file.write(f"{frame_count + i},0,0,0\n")
        else:
            contours, _ = cv2.findContours(ball_binary_heatmaps[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_bounding_box = max([cv2.boundingRect(c) for c in contours], key=lambda r: r[2] * r[3])
            
            predicted_x_center = int(width_ratio * (largest_bounding_box[0] + largest_bounding_box[2] / 2))
            predicted_y_center = int(height_ratio * (largest_bounding_box[1] + largest_bounding_box[3] / 2))

            predicted_points_queue.appendleft((predicted_x_center, predicted_y_center))
            if len(predicted_points_queue) > predicted_points_queue.maxlen:
                predicted_points_queue.pop()

            for point in predicted_points_queue:
                if point is not None:
                    cv2.circle(annotated_frames[i], point, 5, (0, 255, 0), 2)
            cv2.circle(annotated_frames[i], (predicted_x_center, predicted_y_center), 5, (0, 0, 255), -1)

            with open(ball_csv_path, 'a') as csv_file:
                csv_file.write(f"{frame_count+i},{predicted_x_center},{predicted_y_center},1\n")

        # Player processing
        if player_predictions is not None:
            player_heatmaps = (player_predictions > 0.5).float()
            player_binary_heatmaps = (player_heatmaps[0] * 255).byte().numpy()
            if np.amax(player_binary_heatmaps[i]) > 0:
                contours, _ = cv2.findContours(player_binary_heatmaps[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x1 = int(width_ratio * x)
                    y1 = int(height_ratio * y)
                    x2 = int(width_ratio * (x + w))
                    y2 = int(height_ratio * (y + h))
                    cv2.rectangle(annotated_frames[i], (x1, y1), (x2, y2), (255, 0, 0), 2)
                    with open(player_csv_path, 'a') as csv_file:
                        csv_file.write(f"{frame_count+i},{x1},{y1},{x2},{y2}\n")

        video_writer.write(annotated_frames[i])


def main(args):
    """
    Main function to run the prediction script.
    """
    # Unpack arguments
    video_path = args.video_path
    model_weights = args.model_weights
    model_name = args.model_name
    output_dir = args.output_dir
    queue_length = args.queue_length

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = get_model(model_name, INPUT_HEIGHT, INPUT_WIDTH)
    model.load_state_dict(torch.load(model_weights))
    model.to(device)
    model.eval()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video capture
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)

    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up video writer
    output_video_path = os.path.join(output_dir, os.path.basename(video_path))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Set up CSV output
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    ball_csv_path = os.path.join(output_dir, f"{base_name}_ball_predictions.csv")
    player_csv_path = os.path.join(output_dir, f"{base_name}_player_predictions.csv")
    
    with open(ball_csv_path, 'w') as csv_file:
        csv_file.write("frame,x,y,visibility\n")
    if 'TrackNetV5' in model_name:
        with open(player_csv_path, 'w') as csv_file:
            csv_file.write("frame,x1,y1,x2,y2\n")


    # Initialize queues
    frame_queue = queue.Queue()
    predicted_points_queue = deque([None] * queue_length, maxlen=queue_length)

    # Ratios for coordinate conversion
    width_ratio = frame_width / INPUT_WIDTH
    height_ratio = frame_height / INPUT_HEIGHT

    frame_count = 0
    total_inference_time = 0
    num_inferences = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_queue.put(frame)

        if frame_queue.qsize() >= 3:
            frame1 = frame_queue.get()
            frame2 = frame_queue.get()
            frame3 = frame_queue.get()

            (ball_predictions, player_predictions), inference_time = run_model_inference(model, [frame1, frame2, frame3], device)
            total_inference_time += inference_time
            num_inferences += 1

            post_process_predictions(ball_predictions, player_predictions, frame1, frame2, frame3, frame_count, video_writer, ball_csv_path, player_csv_path, width_ratio, height_ratio, predicted_points_queue)
            frame_count += 1

            # Put back frames for overlapping windows
            frame_queue.put(frame2)
            frame_queue.put(frame3)

        print(f"Processing frame {frame_count}/{total_frames}", end='\r')

    # Process remaining frames in the queue
    while frame_queue.qsize() >= 3:
        frame1 = frame_queue.get()
        frame2 = frame_queue.get()
        frame3 = frame_queue.get()
        (ball_predictions, player_predictions), inference_time = run_model_inference(model, [frame1, frame2, frame3], device)
        post_process_predictions(ball_predictions, player_predictions, frame1, frame2, frame3, frame_count, video_writer, ball_csv_path, player_csv_path, width_ratio, height_ratio, predicted_points_queue)
        frame_count += 3
        frame_queue.put(frame2)
        frame_queue.put(frame3)

    while not frame_queue.empty():
        frame = frame_queue.get()
        video_writer.write(frame)


    # Release resources
    video_capture.release()
    video_writer.release()

    print(f"\nPrediction complete. Output saved to {output_dir}")
    if num_inferences > 0:
        print(f"Average inference time: {total_inference_time / num_inferences:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict trajectories on video using a trained model.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the trained model weights (.pth).")
    parser.add_argument("--model_name", type=str, required=True, choices=['Baseline_TrackNetV2', 'TrackNetV4_TypeA', 'TrackNetV4_TypeB', 'TrackNetV5_TypeA', 'TrackNetV5_TypeB'], help="Name of the model to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output video and CSV.")
    parser.add_argument("--queue_length", type=int, default=10, help="Length of the trajectory queue.")
    
    args = parser.parse_args()
    main(args)
