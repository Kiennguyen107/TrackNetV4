#!/usr/bin/env python
"""
Training Script
-------------------------

This script trains a TrackNet model on a specified dataset using configurable parameters.
It supports loading a pretrained model, saving checkpoints, and evaluating performance during training.

Usage:
    python src/train.py --model_name <MODEL> --dataset <DATASET> [options]

Example:
    python src/train.py --model_name Baseline_TrackNetV2 --dataset tennis_game_level_split \
        --batch_size 3 --learning_rate 1.0 --height 288 --width 512 --epochs 30 --tol 4 \
        --work_dir ./models --save_freq 1

Arguments:
    --model_name   : Name of the model to use.
                     Allowed values: Baseline_TrackNetV2.
    --dataset      : Name of the dataset to use.
                     Allowed values: tennis_game_level_split, tennis_clip_level_split, badminton, new_tennis.
    --batch_size   : Batch size for training (default: 3).
    --learning_rate: Learning rate for the optimizer (default: 1.0).
    --height       : Target image height (default: 288).
    --width        : Target image width (default: 512).
    --epochs       : Number of epochs for training (default: 30).
    --tol          : Tolerance for the outcome evaluation (default: 4).
    --model_path   : Path to a pretrained model (.pth) to load before training (optional).
    --work_dir     : Directory to save the trained models (default: "./models").
    --save_freq    : Frequency (in epochs) to save model checkpoints (default: 1).

Note:
    If the default work directory is used, a timestamp will be appended to create a unique directory.
"""

import argparse
import datetime
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from util import get_dataset, outcome, get_model
# from models.TrackNetV2_pt import TrackNetV2 as TrackNetV2_pt
# from models.TrackNetV4_pt import TrackNetV4 as TrackNetV4_pt


def main(args):
    """
    Train the TrackNet model using specified configurations.
    """
    # Unpack arguments
    model_name = args.model_name
    dataset_name = args.dataset
    batch_size = args.batch_size
    height = args.height
    width = args.width
    learning_rate = args.learning_rate
    epochs = args.epochs
    tol = args.tol
    save_freq = args.save_freq
    model_path = args.model_path
    work_dir = args.work_dir

    # If using default work directory, append a timestamp for uniqueness
    if work_dir == "./models":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(work_dir, timestamp)

    # Print all experiment configurations before starting the training
    experiment_config = {
        "model_name": model_name,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "height": height,
        "width": width,
        "epochs": epochs,
        "tol": tol,
        "model_path": model_path,
        "work_dir": work_dir,
        "save_freq": save_freq,
    }
    print("Training Configurations:")
    for key, value in experiment_config.items():
        print(f"  {key}: {value}")

    # Create the work directory if it doesn't exist
    os.makedirs(work_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    train_dataset, val_dataset = get_dataset(dataset_name, height, width)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Load model
    model = get_model(model_name, height, width)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"======== Epoch {epoch + 1} ========")
        for i, (clip_inputs, clip_labels) in enumerate(train_loader):
            clip_inputs = clip_inputs.float().to(device)
            clip_labels = clip_labels.float().to(device)

            # Squeeze the batch dimension added by the DataLoader
            clip_inputs = clip_inputs.squeeze(0)
            clip_labels = clip_labels.squeeze(0)
            
            # print("Clip input shape:", clip_inputs.shape, " Clip label shape:", clip_labels.shape)
            # Manually iterate over the clip's sequences in batches
            for j in range(0, clip_inputs.size(0), batch_size):
                inputs = clip_inputs[j:j+batch_size]
                labels = clip_labels[j:j+batch_size]

                optimizer.zero_grad()

                if 'TrackNetV4' in model_name:
                    outputs, motion_loss = model(inputs)
                    loss = criterion(outputs, labels) + motion_loss
                if 'TrackNetV5' in model_name:
                    ball_outputs, player_outputs, motion_loss = model(inputs)
                    ball_labels = labels[:, :, 0, :, :]
                    player_labels = labels[:, :, 1, :, :]

                    loss = criterion(ball_outputs, ball_labels) + criterion(player_outputs, player_labels) + motion_loss
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            if (i  + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Avg Clip Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        tp, tn, fp, fn = 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                inputs = inputs.squeeze(0)
                labels = labels.squeeze(0)
                for j in range(0, inputs.size(0), batch_size):
                    batch_inputs = inputs[j:j+batch_size]
                    batch_labels = labels[j:j+batch_size]
                    if 'TrackNetV5' in model_name:
                        ball_outputs, player_outputs, _ = model(batch_inputs)
                        ball_labels = batch_labels[:, :, 0, :, :]
                        player_labels = batch_labels[:, :, 1, :, :]
                        val_loss += criterion(ball_outputs, ball_labels).item()
                        val_loss += criterion(player_outputs, player_labels).item()
                        # Calculate TP, TN, FP, FN for outcome (for ball only)
                        preds = (ball_outputs > 0.5).cpu()
                        t, n, p, f = outcome(ball_labels.cpu(), preds, tol)
                        tp += t
                        tn += n
                        fp += p
                        fn += f
                        # Calculate TP, TN, FP, FN for outcome (for player only)
                        preds = (player_outputs > 0.5).cpu()
                        t, n, p, f = outcome(player_labels.cpu(), preds, tol)
                        tp += t
                        tn += n
                        fp += p
                        fn += f
                    elif 'TrackNetV4' in model_name:
                        outputs, _ = model(batch_inputs)
                        val_loss += criterion(outputs, batch_labels).item()
                        preds = (outputs > 0.5).cpu()
                        t, n, p, f = outcome(batch_labels.cpu(), preds, tol)
                        tp += t
                        tn += n
                        fp += p
                        fn += f
                    else:
                        outputs = model(batch_inputs)
                        val_loss += criterion(outputs, batch_labels).item()
                        preds = (outputs > 0.5).cpu()
                        t, n, p, f = outcome(batch_labels.cpu(), preds, tol)
                        tp += t
                        tn += n
                        fp += p
                        fn += f

        val_loss /= len(val_loader.dataset) # Average loss per sample
        print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}')
        print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
        
        end_time = time.time()
        print(f"Time taken for epoch {epoch + 1}: {end_time - start_time:.2f} seconds")
        start_time = end_time
        # Save model
        if (epoch + 1) % save_freq == 0:
            model_save_path = os.path.join(work_dir, f"{model_name}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TrackNet model with configurable parameters."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['Baseline_TrackNetV2', 'TrackNetV4_TypeA', 'TrackNetV4_TypeB', 'TrackNetV5_TypeA', 'TrackNetV5_TypeB'],
        help="Name of the model to use."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['tennis_game_level_split', 'tennis_clip_level_split', 'badminton', 'new_tennis'],
        help="Name of the dataset to use."
    )
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate for the optimizer")
    parser.add_argument("--height", type=int, default=288, help="Target height of the images")
    parser.add_argument("--width", type=int, default=512, help="Target width of the images")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--tol", type=int, default=4, help="Tolerance for the outcome evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model (.pth) to load before training"
    )
    parser.add_argument("--work_dir", type=str, default="./models", help="Directory to save the trained models")
    parser.add_argument("--save_freq", type=int, default=1, help="Frequency (in epochs) to save model checkpoints")
    
    args = parser.parse_args()
    main(args)
