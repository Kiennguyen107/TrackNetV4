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
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import keras.backend as K
from models.TrackNetV4 import MotionPromptLayer, FusionLayerTypeA, FusionLayerTypeB
from constants import HEIGHT, WIDTH
from util import custom_loss, get_model

# Constants
BATCH_SIZE = 1
INPUT_HEIGHT = 288
INPUT_WIDTH = 512

def run_model_inference(model, frames):
    """
    Pre-processes the frames and runs inference on the model.
    """
    input_batch = []
    
    # Preprocess the frames for model input
    for frame in frames:
        resized_frame = array_to_img(frame[..., ::-1]).resize((INPUT_WIDTH, INPUT_HEIGHT))
        frame_array = img_to_array(resized_frame) / 255.0  # Normalize
        input_batch.append(frame_array)
    
    # Stack frames: (3, H, W, C) -> concatenate along channel axis
    input_batch = np.concatenate(input_batch, axis=-1)  # Shape: (H, W, 9)
    input_batch = np.expand_dims(input_batch, axis=0)   # Shape: (1, H, W, 9)

    # Perform prediction
    inference_start_time = time.time()
    predictions = model.predict(input_batch, batch_size=BATCH_SIZE, verbose=1)
    inference_end_time = time.time()

    inference_time = inference_end_time - inference_start_time
    return predictions, inference_time

def post_process_predictions(predictions, frame1, frame2, frame3, frame_count, video_writer, csv_output_path, width_ratio, height_ratio, predicted_points_queue):
    """
    Post-processes the predictions, annotates the video, and saves results.
    
    Args:
        predictions: Predictions from the model.
        frame1, frame2, frame3: Original frames.
        frame_count: Current frame count in the video.
        video_writer: Video writer object for saving frames.
        csv_output_path: Path to the CSV file for saving results.
        width_ratio: Ratio to adjust the predicted points' width.
        height_ratio: Ratio to adjust the predicted points' height.
        predicted_points_queue: Deque storing the predicted points.
    """
    binary_predictions = (predictions > 0.5).astype('float32')
    binary_heatmaps = (binary_predictions[0] * 255).astype('uint8')

    for i, current_frame in enumerate([frame1, frame2, frame3]):
        if np.amax(binary_heatmaps[i]) <= 0:
            with open(csv_output_path, 'a') as csv_file:
                csv_file.write(f"{frame_count},0,0,0\n")
            video_writer.write(current_frame)
        else:
            contours, _ = cv2.findContours(binary_heatmaps[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
            largest_bounding_box = max(bounding_boxes, key=lambda r: r[2] * r[3])
            
            # Calculate predicted center
            predicted_x_center = int(width_ratio * (largest_bounding_box[0] + largest_bounding_box[2] / 2))
            predicted_y_center = int(height_ratio * (largest_bounding_box[1] + largest_bounding_box[3] / 2))

            # Update deque and draw trajectory
            predicted_points_queue.appendleft((predicted_x_center, predicted_y_center))
            predicted_points_queue.pop()

            frame_copy = np.copy(current_frame)
            for point in predicted_points_queue:
                if point is not None:
                    cv2.circle(frame_copy, point, 5, (0, 255, 0), 2)
            cv2.circle(frame_copy, (predicted_x_center, predicted_y_center), 5, (0, 0, 255), -1)
            video_writer.write(frame_copy)

            # Write results to CSV
            with open(csv_output_path, 'a') as csv_file:
                csv_file.write(f"{frame_count},1,{predicted_x_center},{predicted_y_center}\n")

def main(args):
    """
    Main function to handle video processing, model inference, and result saving.
    """
    video_path = args.video_path
    model_weights_path = args.model_weights
    output_dir = args.output_dir
    queue_length = args.queue_length
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect model type from filename
    weights_filename = os.path.basename(model_weights_path).lower()
    if 'type_a' in weights_filename or 'typea' in weights_filename:
        model_name = "TrackNetV4_TypeA"
    elif 'type_b' in weights_filename or 'typeb' in weights_filename:
        model_name = "TrackNetV4_TypeB"
    elif 'v2' in weights_filename or 'baseline' in weights_filename:
        model_name = "Baseline_TrackNetV2"
    else:
        # Default fallback
        model_name = "TrackNetV4_TypeA"
        print(f"Warning: Could not detect model type from filename. Using default: {model_name}")
    
    print(f"Detected model type: {model_name}")
    
    # Initialize the predicted points queue with the specified length
    predicted_points_queue = queue.deque([None] * queue_length)

    # Load model
    if model_weights_path.endswith('.weights.h5'):
        model = get_model(model_name, INPUT_HEIGHT, INPUT_WIDTH)
        model.load_weights(model_weights_path)
    else:
        model = load_model(
            model_weights_path, 
            custom_objects={
                'MotionPromptLayer': MotionPromptLayer,
                'MotionIncorporationLayerV1': MotionIncorporationLayerV1,  # Ensure these are imported or defined
                'MotionIncorporationLayerV2': MotionIncorporationLayerV2,
                'CombineOutputs': CombineOutputs,
                'MotionFramesInput': MotionFramesInput,
                'custom_loss': custom_loss,
            }
        )

    # Read input video and set up output video settings
    video_capture = cv2.VideoCapture(video_path)
    success, frame1 = video_capture.read()
    success, frame2 = video_capture.read()
    success, frame3 = video_capture.read()

    height_ratio = frame1.shape[0] / INPUT_HEIGHT
    width_ratio = frame1.shape[1] / INPUT_WIDTH
    video_size = (int(INPUT_WIDTH * width_ratio), int(INPUT_HEIGHT * height_ratio))
    frames_per_second = int(video_capture.get(cv2.CAP_PROP_FPS))

    if video_path.endswith('avi'):
        video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    elif video_path.endswith('mp4'):
        video_codec = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        print('Error: Video format must be .avi or .mp4')
        sys.exit(1)

    # Output files
    output_video_path = os.path.join(output_dir, os.path.basename(video_path[:-4] + '_predict' + video_path[-4:]))
    csv_output_path = os.path.join(output_dir, os.path.basename(video_path[:-4] + '_predict.csv'))

    with open(csv_output_path, 'w') as csv_file:
        csv_file.write('Frame,Visibility,X,Y\n')
    video_writer = cv2.VideoWriter(output_video_path, video_codec, frames_per_second, video_size)

    # Main loop
    frame_count = 0
    total_inference_time = 0.0
    start_time = time.time()

    while success:
        frames = [frame1, frame2, frame3]
        predictions, inference_time = run_model_inference(model, frames)
        post_process_predictions(predictions, frame1, frame2, frame3, frame_count, video_writer, csv_output_path, width_ratio, height_ratio, predicted_points_queue)
        
        # Accumulate the total inference time
        total_inference_time += inference_time
        frame_count += 1

        # Read next set of frames
        success, frame1 = video_capture.read()
        success, frame2 = video_capture.read()
        success, frame3 = video_capture.read()

    # Clean up resources
    video_writer.release()
    end_time = time.time()

    # Output timing information
    if total_inference_time > 0:
        inference_speed = frame_count / total_inference_time
        print(f'Total model inference time: {total_inference_time:.2f} seconds')
        print(f'Model inference speed: {inference_speed:.2f} frames/sec')

    print(f'Total script runtime: {end_time - start_time:.2f} seconds')
    print('Prediction complete.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict trajectories on a video using a trained model")
    parser.add_argument('--video_path', required=True, help="Path to the video file")
    parser.add_argument('--model_weights', required=True, help="Path to the model weights")
    parser.add_argument('--output_dir', default=os.getcwd(), help="Directory to save output files (default: current working directory)")
    parser.add_argument('--queue_length', type=int, default=5, help="Length of the predicted points queue (default: 5)")
    args = parser.parse_args()
    main(args)
