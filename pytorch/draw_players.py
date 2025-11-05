import cv2
import pandas as pd
import os

def create_video_from_csv(clips_dir, csv_path, output_path, fps=30):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Group by frame for easy access
    grouped = df.groupby("frame")

    # Sort frames
    frames = sorted(grouped.groups.keys())

    # Get first frame to initialize video writer
    first_frame_path = os.path.join(clips_dir, f"{frames[0]}.jpg")
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"First frame not found: {first_frame_path}")
    
    first_img = cv2.imread(first_frame_path)
    h, w, _ = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Colors per player
    colors = {
        1: (255, 0, 0),   # Blue in BGR
        2: (0, 255, 0),   # Green
    }

    # Loop through frames
    for f in frames:
        img_path = os.path.join(clips_dir, f"{f}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: missing frame {f}")
            continue

        rows = grouped.get_group(f)
        for _, row in rows.iterrows():
            player_id = row["player_id"]
            # center of rect
            cx = int((row["x1"] + row["x2"]) / 2)
            cy = int((row["y1"] + row["y2"]) / 2)
            color = colors.get(player_id, (0, 0, 255))  # Default red if unknown player
            cv2.circle(img, (cx, cy), 8, color, -1)

        out.write(img)

    out.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", required=True, help="Directory containing frame images (0.jpg, 1.jpg, ...)")
    parser.add_argument("--csv", required=True, help="CSV file with rects")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--fps", type=int, default=30, help="FPS for output video")
    args = parser.parse_args()

    create_video_from_csv(args.clips_dir, args.csv, args.output, args.fps)
