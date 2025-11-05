import csv
import os
import cv2
import argparse
import math
import numpy as np
from ultralytics import YOLO
from collections import deque


class PlayerTracker:
    """Tracks a single player's trajectory and movement patterns."""

    def __init__(self, initial_position, player_id, history_length=None, exponential_prediction=True): # Allow unlimited history
        self.player_id = player_id
        self.position_history = deque([initial_position], maxlen=history_length)
        self.box_history = deque(maxlen=history_length) # To store bounding boxes
        self.frame_history = deque(maxlen=history_length) # To store frame indices
        self.velocity_history = deque(maxlen=history_length-1 if history_length else None)
        self.confidence_history = deque(maxlen=history_length)
        self.frames_lost = 0
        self.total_movement = 0
        self.is_active = True
        self.exponential_prediction = exponential_prediction

    def update_position(self, new_position, confidence, box, frame_index):
        """Update player position and calculate velocity."""
        if len(self.position_history) > 0:
            velocity = (new_position[0] - self.position_history[-1][0], 
                        new_position[1] - self.position_history[-1][1])
            self.velocity_history.append(velocity)
            
            # Calculate movement distance
            movement = math.sqrt(velocity[0]**2 + velocity[1]**2)
            self.total_movement += movement
            
        self.position_history.append(new_position)
        self.confidence_history.append(confidence)
        self.box_history.append(box)
        self.frame_history.append(frame_index)
        self.frames_lost = 0

    def current_position(self):
        """Return the last known position of the player."""
        return (self.position_history[-1] if self.position_history else None), \
                (self.confidence_history[-1] if self.confidence_history else None), \
                (self.box_history[-1] if self.box_history else None)

    def predict_next_position(self):
        """Predict next position based on velocity history."""
        if len(self.velocity_history) == 0:
            return self.position_history[-1]

        # Convert deque → list for slicing
        vel_hist = list(self.velocity_history)

        if self.exponential_prediction:
            # Take last 50 velocities
            window = vel_hist[-50:]
            n = len(window)
            # Recent items get larger exponent → higher weight
            weights = np.exp(np.linspace(-self.exponential_prediction, 0, n))
            weights /= weights.sum()
            avg_velocity = np.average(window, axis=0, weights=weights)
        else:
            window = vel_hist
            weights = np.array([(i + 1) ** 2 for i in range(len(window))])
            weights = weights / weights.sum()
            avg_velocity = np.average(window, axis=0, weights=weights)

        predicted_pos = (
            self.position_history[-1][0] + avg_velocity[0],
            self.position_history[-1][1] + avg_velocity[1]
        )
        return predicted_pos


        
    def get_average_velocity(self):
        """Get average velocity magnitude over recent frames."""
        if len(self.velocity_history) == 0:
            return 0
        velocities = [math.sqrt(v[0]**2 + v[1]**2) for v in self.velocity_history]
        return sum(velocities) / len(velocities)
        
    def get_movement_consistency(self):
        """Calculate how consistent the movement direction is (0-1, higher = more consistent)."""
        if len(self.velocity_history) < 2:
            return 0.5
            
        # Calculate angle changes between consecutive velocity vectors
        angle_changes = []
        for i in range(1, len(self.velocity_history)):
            v1 = self.velocity_history[i-1]
            v2 = self.velocity_history[i]
            
            # Calculate angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle_changes.append(abs(math.acos(cos_angle)))
                
        if not angle_changes:
            return 0.5
            
        # Lower angle changes = higher consistency
        avg_angle_change = sum(angle_changes) / len(angle_changes)
        consistency = max(0, 1 - (avg_angle_change / math.pi))
        return consistency
        
    def is_likely_player(self, min_avg_velocity=2.0, min_total_movement=10.0):
        """Determine if this tracker represents an active tennis player."""
        avg_vel = self.get_average_velocity()
        movement_per_frame = self.total_movement / max(1, len(self.position_history))
        
        return (avg_vel >= min_avg_velocity or 
                movement_per_frame >= min_total_movement or
                self.get_movement_consistency() > 0.3)

def calculate_distance(box1_center, box2_center):
    """Calculate Euclidean distance between two bounding box centers."""
    return math.sqrt((box1_center[0] - box2_center[0])**2 + (box1_center[1] - box2_center[1])**2)

def get_box_center(box):
    """Get the center point of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def process_frame(image_path, model, frame_index, player_trackers=None, max_distance=30, max_lost_frames=5, calibration_mode=False, all_trackers=None, exponential_prediction=True):
    """
    Detects players in a single image using trajectory tracking for smooth capture.
    Distinguishes between active players and stationary entities like ball keepers.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, player_trackers, all_trackers

    results = model(image_path,verbose=False)
    
    # Collect all person detections with their confidence and bounding box
    person_detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            if class_name == 'person':
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center = get_box_center((x1, y1, x2, y2))
                person_detections.append((confidence, (x1, y1, x2, y2), center))

    if calibration_mode:
        # In calibration mode, we track all detected persons
        if all_trackers is None:
            all_trackers = []

        used_detections = set()
        # Match detections to existing trackers
        for tracker in all_trackers:
            best_detection = None
            min_dist = max_distance + 1
            best_idx = -1
            for idx, (conf, box, center) in enumerate(person_detections):
                if idx in used_detections:
                    continue
                dist = calculate_distance(tracker.position_history[-1], center)
                if dist < min_dist:
                    min_dist = dist
                    best_detection = (conf, box, center)
                    best_idx = idx
            
            if best_detection is not None and min_dist <= max_distance:
                conf, box, center = best_detection
                tracker.update_position(center, conf, box, frame_index)
                used_detections.add(best_idx)
            else:
                tracker.frames_lost += 1

        # Add new trackers for unmatched detections
        for idx, (conf, box, center) in enumerate(person_detections):
            if idx not in used_detections:
                new_tracker = PlayerTracker(center, len(all_trackers), exponential_prediction=exponential_prediction)
                new_tracker.update_position(center, conf, box, frame_index)
                all_trackers.append(new_tracker)
        
        # Draw all tracked objects during calibration
        for tracker in all_trackers:
            if tracker.frames_lost == 0: # Only draw active trackers
                x, y = tracker.position_history[-1]
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), -1)
                cv2.putText(img, f"T{tracker.player_id}", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return img, player_trackers, all_trackers

    if not person_detections:
        # Increment lost frames for all trackers
        if player_trackers:
            for tracker in player_trackers:
                tracker.frames_lost += 1
        return img, player_trackers, all_trackers

    # Initialize trackers on first frame (This part is now handled by calibration)
    if player_trackers is None:
        # This should not be called if calibration is done properly.
        # Fallback to original method if called without calibration.
        person_detections.sort(reverse=True, key=lambda x: x[0])
        player_trackers = []
        for i, (conf, box, center) in enumerate(person_detections[:2]):
            tracker = PlayerTracker(center, i, exponential_prediction=exponential_prediction)
            tracker.update_position(center, conf, box, frame_index)
            player_trackers.append(tracker)
        selected_detections = [(conf, box) for conf, box, _ in person_detections[:2]]
    else:
        # Update existing trackers with trajectory prediction
        selected_detections = []
        used_detections = set()
        
        # Remove trackers that have been lost for too long
        player_trackers = [t for t in player_trackers if t.frames_lost < max_lost_frames]
        
        # Match detections to existing trackers
        for tracker in player_trackers:
            # Predict next position and cap within frame boundaries
            predicted_pos = tracker.predict_next_position()

            h, w, _ = img.shape
            px = max(0, min(int(predicted_pos[0]), w - 1))
            py = max(0, min(int(predicted_pos[1]), h - 1))
            predicted_pos = (px, py)
            best_detection = None
            best_score = -1
            best_idx = -1
            
            for idx, (confidence, box, center) in enumerate(person_detections):
                if idx in used_detections:
                    continue
                    
                distance_to_current = calculate_distance(tracker.position_history[-1], center)
                distance_to_predicted = calculate_distance(predicted_pos, center)
                
                # Use the minimum of both distances for better tracking
                effective_distance = min(distance_to_current, distance_to_predicted)
                
                if effective_distance <= max_distance:
                    # Trajectory-based scoring
                    movement_velocity = calculate_distance((0, 0), 
                        (center[0] - tracker.position_history[-1][0], 
                         center[1] - tracker.position_history[-1][1]))
                    
                    # Score components:
                    confidence_score = confidence * 0.3
                    distance_score = (1 - effective_distance / max_distance) * 0.2
                    prediction_score = (1 - distance_to_predicted / max_distance) * 0.2
                    movement_score = min(1.0, movement_velocity / 15.0) * 0.2  # Reward reasonable movement
                    consistency_score = tracker.get_movement_consistency() * 0.1
                    
                    combined_score = (confidence_score + distance_score + 
                                    prediction_score + movement_score + consistency_score)

                    if combined_score > best_score:
                        best_score = combined_score
                        best_detection = (confidence, box, center)
                        best_idx = idx
            
            if best_detection is not None:
                conf, box, center = best_detection
                tracker.update_position(center, conf, box, frame_index)
                selected_detections.append((conf, (box[0], box[1], box[2], box[3])))
                used_detections.add(best_idx)
            else:
                print(f"Tracker {tracker.player_id} lost in frame {frame_index}")
                tracker.frames_lost += 1
                # use old detection into tracker
                if len(tracker.position_history) > 0:
                    predicted_pos = tracker.predict_next_position()
                    last_pos, last_conf, last_box = tracker.current_position()
                    # last_pos, last_conf, last_box = tracker.current_position()
                    tracker.update_position(predicted_pos,
                                            last_conf,
                                            last_box, frame_index)
                    selected_detections.append((last_conf, (last_box[0], last_box[1], last_box[2], last_box[3])))
                else:
                    # No history, cannot use old detection
                    pass
        
        # # Add new trackers for unmatched high-confidence detections that show movement potential
        # remaining_detections = [(conf, box, center) for idx, (conf, box, center) in enumerate(person_detections) 
        #                       if idx not in used_detections]
        
        # if len(selected_detections) < 2 and remaining_detections:
        #     # Filter remaining detections for movement potential
        #     remaining_detections.sort(reverse=True, key=lambda x: x[0])
            
        #     for conf, box, center in remaining_detections:
        #         if len(selected_detections) >= 2:
        #             break
                    
        #         # Check if this detection is likely a moving player
        #         is_potential_player = True
                
        #         # Check distance from existing stationary positions (if any)
        #         for tracker in player_trackers:
        #             if (tracker.get_average_velocity() < 1.0 and  # Stationary tracker
        #                 calculate_distance(center, tracker.position_history[-1]) < 10):  # Very close
        #                 is_potential_player = False
        #                 break
                
        #         if is_potential_player:
        #             # Create new tracker
        #             new_tracker = PlayerTracker(center, len(player_trackers))
        #             new_tracker.update_position(center, conf, box, frame_index)
        #             player_trackers.append(new_tracker)
        #             selected_detections.append((conf, (box[0], box[1], box[2], box[3])))
        
        # # Filter out trackers that are clearly not players (stationary for too long)
        # active_trackers = []
        # active_detections = []
        
        # for i, tracker in enumerate(player_trackers):
        #     if i < len(selected_detections) and tracker.is_likely_player():
        #         active_trackers.append(tracker)
        #         active_detections.append(selected_detections[i])
        
        # # If we filtered too many, add back the most confident ones
        # if len(active_detections) < min(2, len(selected_detections)):
        #     for i, tracker in enumerate(player_trackers):
        #         if len(active_detections) >= 2:
        #             break
        #         if i < len(selected_detections) and tracker not in active_trackers:
        #             active_trackers.append(tracker)
        #             active_detections.append(selected_detections[i])
        
        # player_trackers = active_trackers
        # selected_detections = active_detections

    # Draw bounding boxes and trajectory trails
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Different colors for different players
    
    for i, (confidence, (x1, y1, x2, y2)) in enumerate(selected_detections):
        color = colors[i % len(colors)]
        
        # Draw bounding box
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw trajectory trail if tracker exists
        if i < len(player_trackers) and len(player_trackers[i].position_history) > 1:
            points = list(player_trackers[i].position_history)
            for j in range(1, len(points)):
                # Fade the trail (older points are more transparent)
                alpha = j / len(points)
                thickness_trail = max(1, int(3 * alpha))
                cv2.line(img, 
                        (int(points[j-1][0]), int(points[j-1][1])), 
                        (int(points[j][0]), int(points[j][1])), 
                        color, thickness_trail)

        # Put label and confidence with tracker info
        if i < len(player_trackers):
            avg_vel = player_trackers[i].get_average_velocity()
            label = f"Player {i+1}: {confidence:.2f} (v:{avg_vel:.1f})"
        else:
            label = f"Player: {confidence:.2f}"
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Position text slightly above the bounding box
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10
        
        # Draw background for text for better readability
        cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), 
                     (text_x + text_size[0] + 5, text_y + 5), color, -1)
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return img, player_trackers, all_trackers

def create_video_from_frames(input_folder, output_path, csv_path, fps=24, max_distance=30, max_lost_frames=5, device="auto", calibration_frames=10, exponential_prediction=True):
    """
    Processes all images in a folder, saves them as a video, and sorts them by name.
    Uses trajectory tracking for smooth player capture and filtering of stationary entities.
    """
    # Load the pre-trained YOLOv11 model once and move to GPU if available
    model = YOLO('yolo11x.pt')
    print("Exponential prediction is", "enabled" if exponential_prediction else "disabled")
    # Check for GPU availability and use it
    import torch
    
    if device == "auto":
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Auto-selected GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = 'cpu'
            print("GPU not available, using CPU")
    elif device == "cuda":
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    else:  # device == "cpu"
        print("Using CPU as requested")
    
    # Move model to the selected device
    model.to(device)

    # Get all image file paths and sort them alphabetically to maintain video order
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])

    if not image_files:
        print(f"No images found in the specified folder: {input_folder}")
        return

    # Adjust calibration frames for short videos
    total_frames = len(image_files)
    if total_frames < calibration_frames:
        new_calibration_frames = total_frames * 2 // 3
        print(f"Warning: Video has only {total_frames} frames, which is less than calibration_frames={calibration_frames}.")
        print(f"Adjusting calibration_frames to {new_calibration_frames}.")
        calibration_frames = new_calibration_frames

    # --- Calibration Phase ---
    print(f"--- Starting calibration for the first {calibration_frames} frames ---")
    all_trackers = []
    calibration_image_files = image_files[:calibration_frames]

    # Get video dimensions from the first frame
    first_frame_path = os.path.join(input_folder, image_files[0])
    temp_img = cv2.imread(first_frame_path)
    if temp_img is None:
        print("Could not read the first frame to get video dimensions. Exiting.")
        return
    height, width, layers = temp_img.shape
    del temp_img

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 video
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, filename in enumerate(calibration_image_files):
        image_path = os.path.join(input_folder, filename)
        processed_frame, _, all_trackers = process_frame(
            image_path, model, i, None, max_distance, max_lost_frames, 
            calibration_mode=True, all_trackers=all_trackers, exponential_prediction=exponential_prediction
        )
        if processed_frame is not None:
            # Optionally write calibration frames to video to see what's happening
            video_writer.write(processed_frame)
            print(f"Calibration frame {i+1}/{len(calibration_image_files)} - Total trackers: {len(all_trackers)}")
        else:
            print(f"Skipping calibration frame {filename} due to an error.")

    # After calibration, select the top 2 players based on total movement
    if not all_trackers:
        print("No persons were tracked during calibration. Exiting.")
        video_writer.release()
        return

    all_trackers.sort(key=lambda t: t.total_movement, reverse=True)
    player_trackers = all_trackers[:2]

    print("--- Calibration finished ---")
    if player_trackers:
        print("Selected players based on movement:")
        for i, tracker in enumerate(player_trackers):
            print(f"  Player {i+1}: Tracker ID {tracker.player_id}, Total Movement: {tracker.total_movement:.2f} pixels")
    else:
        print("Could not select any players from calibration.")


    # --- Main Processing Phase ---
    remaining_image_files = image_files[calibration_frames:]
    print(f"\nProcessing remaining {len(remaining_image_files)} images with selected players...")
    print(f"Maximum tracking distance: {max_distance} pixels")
    print(f"Maximum lost frames before dropping tracker: {max_lost_frames}")
    print(f"Using device: {device}")
    
    # Process remaining frames
    for i, filename in enumerate(remaining_image_files, start=calibration_frames):
        image_path = os.path.join(input_folder, filename)
        processed_frame, player_trackers, _ = process_frame(
            image_path, model, i, player_trackers, max_distance, max_lost_frames,
            calibration_mode=False, exponential_prediction=exponential_prediction
        )
        if processed_frame is not None:
            video_writer.write(processed_frame)
            
            # Print progress and tracker status every 30 frames
            if i % 30 == 0:
                active_trackers = len([t for t in player_trackers if t.frames_lost == 0]) if player_trackers else 0
                print(f"Frame {i}/{len(image_files)-1} - Active trackers: {active_trackers}")
                # Print GPU memory usage if using GPU
                if device == 'cuda':
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_cached = torch.cuda.memory_reserved() / 1024**3
                    print(f"GPU Memory - Used: {memory_used:.2f}GB, Cached: {memory_cached:.2f}GB")
        else:
            print(f"Skipping frame {filename} due to an error.")

    # Release the VideoWriter object
    video_writer.release()
    
    # --- Save trajectories to CSV ---
    if csv_path and player_trackers:
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['frame', 'player_id', 'x1', 'y1', 'x2', 'y2'])
            
            for i, tracker in enumerate(player_trackers):
                player_id = i + 1
                for frame_index, box in zip(tracker.frame_history, tracker.box_history):
                    x1, y1, x2, y2 = box
                    csv_writer.writerow([frame_index, player_id, x1, y1, x2, y2])
        print(f"Player trajectories saved to {csv_path}")

    # Clear GPU cache if using GPU
    if device == 'cuda':
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    
    # Print final trajectory statistics
    if player_trackers:
        print(f"\nFinal trajectory statistics:")
        for i, tracker in enumerate(player_trackers):
            avg_vel = tracker.get_average_velocity()
            consistency = tracker.get_movement_consistency()
            total_frames = len(tracker.position_history)
            print(f"Player {i+1}: Avg velocity: {avg_vel:.1f}px/frame, "
                  f"Movement consistency: {consistency:.2f}, "
                  f"Tracked for {total_frames} frames")
    
    print(f"Video saved successfully to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a folder of images with trajectory-based player detection and save to a video.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing image frames.")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Path to save the output video file.")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to save the player trajectories CSV file.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the output video.")
    parser.add_argument("--max_distance", type=int, default=25, help="Maximum tracking distance in pixels between frames.")
    parser.add_argument("--max_lost_frames", type=int, default=10, help="Maximum frames a tracker can be lost before being dropped.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: 'cuda', 'cpu', or 'auto' (default: auto)")
    parser.add_argument("--calibration_frames", type=int, default=30, help="Number of initial frames to use for player selection calibration.")
    parser.add_argument("--exponential_prediction", type=float, default=1, help="Whether to use exponential prediction for player tracking.")

    args = parser.parse_args()
    
    # Override device selection if specified
    device_to_use = args.device
    if args.device != "auto":
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device_to_use = "cpu"
        elif args.device == "cuda":
            print(f"Forcing GPU usage")
        elif args.device == "cpu":
            print(f"Forcing CPU usage")

    create_video_from_frames(args.input_folder, args.output_path, args.csv_path, args.fps, args.max_distance, args.max_lost_frames, device_to_use, args.calibration_frames, args.exponential_prediction)
