#!/usr/bin/env python
"""
PyTorch Training Script for TrackNetV4
-------------------------

This script trains a TrackNetV4 model using PyTorch with:
- Direct loading from preprocessed .npy files
- GPU memory optimization
- Checkpoint system for resuming interrupted training
- Automatic model saving

Usage:
    python train_pytorch.py --model_name TrackNetV4_TypeA --dataset tennis_clip_level_split \
        --batch_size 2 --learning_rate 0.001 --epochs 30
"""

import argparse
import datetime
import os
import gc
import json
import math
import numpy as np
import cv2
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


# ============================================================================
# CONSTANTS - Update these paths according to your setup
# ============================================================================

TENNIS_DATASET_ROOT = "/kaggle/input/tttracking-dataset"  # Update this
BADMINTON_DATASET_ROOT = "./data/badminton"  # Update this
NEW_TENNIS_DATASET_ROOT = "./data/new_tennis"  # Update this
PROCESSED_DATA_DIR = ""  # Folder name containing processed .npy files

WIDTH = 512
HEIGHT = 288


# ============================================================================
# PYTORCH DATASET FOR LOADING .NPY FILES
# ============================================================================

class NpyDataset(Dataset):
    """
    PyTorch Dataset that loads preprocessed .npy files directly.
    Expects files named: x_data_1.npy, y_data_1.npy, x_data_2.npy, y_data_2.npy, etc.
    """
    def __init__(self, processed_folder):
        """
        Args:
            processed_folder: Path to folder containing x_data_*.npy and y_data_*.npy files
        """
        self.processed_folder = processed_folder
        
        # Find all x_data files
        x_files = sorted(glob(os.path.join(processed_folder, "x_data_*.npy")))
        
        if not x_files:
            raise ValueError(f"No .npy files found in {processed_folder}")
        
        # Extract indices from filenames
        self.indices = []
        for x_file in x_files:
            basename = os.path.basename(x_file)
            # Extract number from "x_data_123.npy"
            idx = int(basename.replace("x_data_", "").replace(".npy", ""))
            self.indices.append(idx)
        
        self.indices = sorted(self.indices)
        print(f"✓ Found {len(self.indices)} .npy file pairs in {processed_folder}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            x: (N, 9, H, W) - N sequences of 3 concatenated frames
            y: (N, 3, H, W) - N heatmaps for 3 frames
        """
        file_idx = self.indices[idx]
        
        # Load data
        x_path = os.path.join(self.processed_folder, f'x_data_{file_idx}.npy')
        y_path = os.path.join(self.processed_folder, f'y_data_{file_idx}.npy')
        
        x = np.load(x_path)  # Shape: (N, 9, H, W)
        y = np.load(y_path)  # Shape: (N, 3, H, W)
        
        # Convert to PyTorch tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        return x, y


def get_dataset_path(dataset_name, mode):
    """
    Get the path to processed .npy files based on dataset name and mode.
    
    Args:
        dataset_name: Name of dataset (tennis_game_level_split, tennis_clip_level_split, etc.)
        mode: "train" or "test"
    
    Returns:
        Path to processed folder
    """
    if dataset_name == "tennis_game_level_split":
        return os.path.join(TENNIS_DATASET_ROOT, PROCESSED_DATA_DIR, "game_level", mode)
    elif dataset_name == "tennis_clip_level_split":
        return os.path.join(TENNIS_DATASET_ROOT, PROCESSED_DATA_DIR, "clip_level", mode)
    elif dataset_name == "new_tennis":
        return os.path.join(NEW_TENNIS_DATASET_ROOT, PROCESSED_DATA_DIR, mode)
    elif dataset_name == "badminton":
        return os.path.join(BADMINTON_DATASET_ROOT, PROCESSED_DATA_DIR, mode)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def power_normalization(input_tensor, a, b):
    """Power normalization function"""
    return 1 / (1 + torch.exp(-(5 / (0.45 * torch.abs(torch.tanh(a)) + 1e-1)) * 
                               (torch.abs(input_tensor) - 0.6 * torch.tanh(b))))


class MotionPromptLayer(nn.Module):
    """
    Motion Prompt Layer that generates attention maps from video sequences.
    """
    def __init__(self, penalty_weight=0.0):
        super(MotionPromptLayer, self).__init__()
        
        # Color to grayscale weights (RGB to Gray)
        self.register_buffer('gray_scale', torch.tensor([0.299, 0.587, 0.114]))
        
        # Power normalization parameters
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))
        
        # Temporal attention variation regularization parameter
        self.lambda1 = penalty_weight
        
    def forward(self, video_seq):
        """
        Args:
            video_seq: (B, T, C, H, W) tensor - 3 frames of 3 channels each
        Returns:
            attention_map: (B, T-1, H, W) - motion attention between consecutive frames
            loss: regularization loss
        """
        loss = torch.tensor(0.0, device=video_seq.device)
        
        # Normalize back to [0, 1]
        norm_seq = video_seq * 0.225 + 0.45
        
        # Convert to grayscale: (B, T, C, H, W) -> (B, T, H, W)
        grayscale_video_seq = torch.einsum('btchw,c->bthw', norm_seq, self.gray_scale)
        
        # Frame difference: (B, T-1, H, W)
        B, T, H, W = grayscale_video_seq.shape
        frame_diff = grayscale_video_seq[:, 1:] - grayscale_video_seq[:, :-1]
        
        # Power normalization to get attention map
        attention_map = power_normalization(frame_diff, self.a, self.b)
        
        # Temporal attention variation regularization (only during training)
        if self.training and self.lambda1 > 0:
            norm_attention = attention_map.unsqueeze(2)  # (B, T-1, 1, H, W)
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
            temporal_loss = torch.sum(temp_diff ** 2) / (H * W * (T - 2) * B)
            loss = self.lambda1 * temporal_loss
            
        return attention_map, loss


class FusionLayerTypeA(nn.Module):
    """Fusion layer that incorporates motion using attention maps - version A"""
    def forward(self, feature_map, attention_map):
        """
        Args:
            feature_map: (B, 3, H, W)
            attention_map: (B, 2, H, W)
        Returns:
            (B, 3, H, W)
        """
        output_1 = feature_map[:, 0:1, :, :]
        output_2 = feature_map[:, 1:2, :, :] * attention_map[:, 0:1, :, :]
        output_3 = feature_map[:, 2:3, :, :] * attention_map[:, 1:2, :, :]
        
        return torch.cat([output_1, output_2, output_3], dim=1)


class FusionLayerTypeB(nn.Module):
    """Fusion layer that incorporates motion using attention maps - version B"""
    def forward(self, feature_map, attention_map):
        """
        Args:
            feature_map: (B, 3, H, W)
            attention_map: (B, 2, H, W)
        Returns:
            (B, 3, H, W)
        """
        output_1 = feature_map[:, 0:1, :, :] * attention_map[:, 0:1, :, :]
        output_2 = feature_map[:, 1:2, :, :] * ((attention_map[:, 0:1, :, :] + attention_map[:, 1:2, :, :]) / 2)
        output_3 = feature_map[:, 2:3, :, :] * attention_map[:, 1:2, :, :]
        
        return torch.cat([output_1, output_2, output_3], dim=1)


class TrackNetV4(nn.Module):
    """
    TrackNetV4 model with motion prompts using U-Net architecture.
    
    Input: (B, 9, H, W) - 3 consecutive frames concatenated (each frame has 3 channels)
    Output: (B, 3, H, W) - heatmap for each frame
    """
    def __init__(self, input_height, input_width, fusion_type="TypeA"):
        super(TrackNetV4, self).__init__()
        
        # Motion prompt layer
        self.motion_prompt = MotionPromptLayer(penalty_weight=0.0)
        
        # Fusion layer selection
        if fusion_type == "TypeA":
            self.fusion_layer = FusionLayerTypeA()
        elif fusion_type == "TypeB":
            self.fusion_layer = FusionLayerTypeB()
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Encoder
        # Layer 1-2
        self.conv1_1 = nn.Conv2d(9, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        # Layer 4-5
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        # Layer 7-9
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        
        # Layer 11-13
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        
        # Decoder
        # Layer 15-17
        self.conv5_1 = nn.Conv2d(512 + 256, 256, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.conv5_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(256)
        
        # Layer 19-20
        self.conv6_1 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(128)
        self.conv6_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(128)
        
        # Layer 22-23
        self.conv7_1 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(64)
        self.conv7_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(64)
        
        # Layer 24 - Final output
        self.conv_final = nn.Conv2d(64, 3, 1, padding=0)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        """
        Args:
            x: (B, 9, H, W) - 3 consecutive RGB frames concatenated
        Returns:
            output: (B, 3, H, W) - heatmaps
            motion_loss: regularization loss from motion prompt
        """
        # Reshape for motion prompt: (B, 9, H, W) -> (B, 3, 3, H, W)
        B, _, H, W = x.shape
        motion_input = x.view(B, 3, 3, H, W)
        
        # Get motion attention maps: (B, 2, H, W)
        residual_maps, motion_loss = self.motion_prompt(motion_input)
        
        # Encoder path
        # Block 1
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
        
        # Block 2
        x2 = self.pool(x1)
        x2 = F.relu(self.bn2_1(self.conv2_1(x2)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        
        # Block 3
        x3 = self.pool(x2)
        x3 = F.relu(self.bn3_1(self.conv3_1(x3)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = F.relu(self.bn3_3(self.conv3_3(x3)))
        
        # Block 4 (bottleneck)
        x4 = self.pool(x3)
        x4 = F.relu(self.bn4_1(self.conv4_1(x4)))
        x4 = F.relu(self.bn4_2(self.conv4_2(x4)))
        x4 = F.relu(self.bn4_3(self.conv4_3(x4)))
        
        # Decoder path with skip connections
        # Block 5
        x = self.upsample(x4)
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        
        # Block 6
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.bn6_1(self.conv6_1(x)))
        x = F.relu(self.bn6_2(self.conv6_2(x)))
        
        # Block 7
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.bn7_1(self.conv7_1(x)))
        x = F.relu(self.bn7_2(self.conv7_2(x)))
        
        # Final output: (B, 3, H, W)
        x = self.conv_final(x)
        
        # Apply fusion with motion attention
        x = self.fusion_layer(x, residual_maps)
        
        # Sigmoid activation
        x = torch.sigmoid(x)
        
        return x, motion_loss


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class CustomLoss(nn.Module):
    """Custom focal loss for TrackNet training"""
    def __init__(self):
        super(CustomLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, 3, H, W)
            y_true: (B, 3, H, W)
        """
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        
        loss = -1 * ((1 - y_pred) ** 2 * y_true * torch.log(y_pred) +
                     y_pred ** 2 * (1 - y_true) * torch.log(1 - y_pred))
        
        return loss.mean()


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def outcome(y_pred, y_true, tol):
    """
    Calculate TP, TN, FP1, FP2, FN for evaluation.
    
    Args:
        y_pred: (B, 3, H, W) numpy array
        y_true: (B, 3, H, W) numpy array
        tol: tolerance in pixels
    """
    n = y_pred.shape[0]
    TP = TN = FP1 = FP2 = FN = 0
    
    for i in range(n):
        for j in range(3):
            pred_max = np.amax(y_pred[i, j])
            true_max = np.amax(y_true[i, j])
            
            if pred_max == 0 and true_max == 0:
                TN += 1
            elif pred_max > 0 and true_max == 0:
                FP2 += 1
            elif pred_max == 0 and true_max > 0:
                FN += 1
            elif pred_max > 0 and true_max > 0:
                h_pred = (y_pred[i, j] * 255).astype('uint8')
                h_true = (y_true[i, j] * 255).astype('uint8')
                
                # Find center of predicted ball
                cnts, _ = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts) == 0:
                    FN += 1
                    continue
                    
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = max(range(len(rects)), key=lambda k: rects[k][2] * rects[k][3])
                target = rects[max_area_idx]
                cx_pred = int(target[0] + target[2] / 2)
                cy_pred = int(target[1] + target[3] / 2)
                
                # Find center of true ball
                cnts, _ = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts) == 0:
                    FP2 += 1
                    continue
                    
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = max(range(len(rects)), key=lambda k: rects[k][2] * rects[k][3])
                target = rects[max_area_idx]
                cx_true = int(target[0] + target[2] / 2)
                cy_true = int(target[1] + target[3] / 2)
                
                dist = math.sqrt((cx_pred - cx_true) ** 2 + (cy_pred - cy_true) ** 2)
                
                if dist > tol:
                    FP1 += 1
                else:
                    TP += 1
                    
    return TP, TN, FP1, FP2, FN


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(work_dir, epoch, batch_idx, model, optimizer, config):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'timestamp': str(datetime.datetime.now())
    }
    
    checkpoint_path = os.path.join(work_dir, 'training_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: Epoch {epoch + 1}, Batch {batch_idx}")


def load_checkpoint(work_dir):
    """Load training checkpoint if exists"""
    checkpoint_path = os.path.join(work_dir, 'training_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location='cpu')
    return None


def find_latest_model(work_dir):
    """Find the latest saved model in work_dir"""
    model_files = glob(os.path.join(work_dir, 'model_epoch_*.pth'))
    if not model_files:
        return None
    
    epochs = []
    for f in model_files:
        try:
            epoch_num = int(f.split('model_epoch_')[-1].replace('.pth', ''))
            epochs.append((epoch_num, f))
        except:
            continue
    
    if epochs:
        latest = max(epochs, key=lambda x: x[0])
        return latest[1], latest[0]
    return None


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(args):
    """Train the TrackNetV4 model using PyTorch"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔥 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
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
    work_dir = args.work_dir
    resume = args.resume
    
    # Create work directory
    if work_dir == "./models" and not resume:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(work_dir, timestamp)
    
    os.makedirs(work_dir, exist_ok=True)
    
    # Experiment configuration
    experiment_config = {
        "model_name": model_name,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "height": height,
        "width": width,
        "epochs": epochs,
        "tol": tol,
        "work_dir": work_dir,
        "save_freq": save_freq,
    }
    
    # Print configuration
    print("\n" + "="*60)
    print("Training Configurations:")
    for key, value in experiment_config.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Initialize model
    fusion_type = model_name.split('_')[-1]  # TypeA or TypeB
    model = TrackNetV4(height, width, fusion_type=fusion_type)
    model = model.to(device)
    
    print(f"✓ Model created: {model_name}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss and optimizer
    criterion = CustomLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    
    # Resume from checkpoint if needed
    start_epoch = 0
    start_batch = 0
    
    if resume:
        checkpoint = load_checkpoint("/kaggle/input/gitmodel")
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch_idx']
            print(f"\n📌 RESUMING FROM CHECKPOINT")
            print(f"   Epoch: {start_epoch + 1}/{epochs}")
            print(f"   Batch: {start_batch}")
            print(f"   Timestamp: {checkpoint['timestamp']}\n")
        else:
            print("\n⚠️  No checkpoint found, starting fresh training...\n")
    
    # Load dataset
    print("Loading training dataset...")
    try:
        processed_folder = get_dataset_path(dataset_name, "train")
        dataset_train = NpyDataset(processed_folder)
        print(f"✓ Dataset loaded from: {processed_folder}\n")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"\n💡 Make sure:")
        print(f"   1. Update TENNIS_DATASET_ROOT, BADMINTON_DATASET_ROOT paths in the script")
        print(f"   2. The processed .npy files exist in the correct location")
        print(f"   3. Files are named: x_data_1.npy, y_data_1.npy, etc.\n")
        return
    
    # Note: We don't use DataLoader because each .npy file contains multiple sequences
    # Instead, we iterate through the dataset directly
    
    # Training loop
    print("Starting training...\n")
    
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        model.train()
        epoch_loss = 0.0
        
        # Iterate through each .npy file
        for batch_idx in range(len(dataset_train)):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and batch_idx < start_batch:
                print(f"⏭️  Skipping batch {batch_idx + 1}/{len(dataset_train)}")
                continue
            
            try:
                # Load data from .npy file
                # x_data: (N, 9, H, W), y_data: (N, 3, H, W)
                x_data, y_data = dataset_train[batch_idx]
                
                # Move to device
                x_data = x_data.to(device)
                y_data = y_data.to(device)
                
                # Split into mini-batches if needed
                num_sequences = x_data.shape[0]
                num_batches = (num_sequences + batch_size - 1) // batch_size
                
                batch_losses = []
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_sequences)
                    
                    x_batch = x_data[start_idx:end_idx]
                    y_batch = y_data[start_idx:end_idx]
                    
                    # Forward pass
                    y_pred, motion_loss = model(x_batch)
                    
                    # Calculate loss
                    main_loss = criterion(y_pred, y_batch)
                    total_loss = main_loss + motion_loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    batch_losses.append(total_loss.item())
                
                avg_loss = np.mean(batch_losses)
                epoch_loss += avg_loss
                
                print(f"File {batch_idx + 1}/{len(dataset_train)}: "
                      f"{num_sequences} sequences, "
                      f"Loss = {avg_loss:.6f}", flush=True)
                
                # Save checkpoint after each file
                save_checkpoint(work_dir, epoch, batch_idx + 1, model, optimizer, experiment_config)
                
            except KeyboardInterrupt:
                print(f"\n\n⚠️  TRAINING INTERRUPTED BY USER")
                print(f"✓ Progress saved at Epoch {epoch + 1}, File {batch_idx + 1}")
                print(f"▶️  To resume, run with --resume --work_dir {work_dir}\n")
                return
            
            except Exception as e:
                print(f"❌ ERROR in file {batch_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                print(f"⏩ Continuing to next file...\n", flush=True)
            
            finally:
                # Clear memory
                if 'x_data' in locals():
                    del x_data
                if 'y_data' in locals():
                    del y_data
                if 'x_batch' in locals():
                    del x_batch
                if 'y_batch' in locals():
                    del y_batch
                if 'y_pred' in locals():
                    del y_pred
                    
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Reset start_batch for next epoch
        start_batch = 0
        
        # Print epoch statistics
        avg_epoch_loss = epoch_loss / len(dataset_train) if len(dataset_train) > 0 else 0
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.6f}")
        print(f"{'='*60}\n")
        
        # Evaluate on training set if not disabled
        if not args.disable_eval_on_train:
            print(f"Evaluating on training set...")
            model.eval()
            
            TP = TN = FP1 = FP2 = FN = 0
            
            with torch.no_grad():
                for eval_idx in range(len(dataset_train)):
                    if eval_idx % 10 == 0:
                        print(f"  Evaluated {eval_idx}/{len(dataset_train)} files...", flush=True)
                    
                    try:
                        x_data, y_data = dataset_train[eval_idx]
                        x_data = x_data.to(device)
                        
                        # Process in batches
                        num_sequences = x_data.shape[0]
                        y_pred_list = []
                        
                        for i in range(0, num_sequences, batch_size):
                            end_idx = min(i + batch_size, num_sequences)
                            x_batch = x_data[i:end_idx]
                            
                            y_pred_batch, _ = model(x_batch)
                            y_pred_list.append(y_pred_batch.cpu())
                        
                        y_pred = torch.cat(y_pred_list, dim=0)
                        
                        # Threshold predictions
                        y_pred_np = (y_pred.numpy() > 0.5).astype('float32')
                        y_true_np = y_data.numpy()
                        
                        # Calculate metrics
                        tp, tn, fp1, fp2, fn = outcome(y_pred_np, y_true_np, tol)
                        TP += tp
                        TN += tn
                        FP1 += fp1
                        FP2 += fp2
                        FN += fn
                        
                    except Exception as e:
                        print(f"⚠️  Eval error on file {eval_idx + 1}: {e}")
                    
                    finally:
                        if 'x_data' in locals():
                            del x_data
                        if 'y_data' in locals():
                            del y_data
                        if 'y_pred' in locals():
                            del y_pred
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
            
            print(f"\nEpoch {epoch + 1} Evaluation Results:")
            print(f"  TP={TP}, TN={TN}, FP1={FP1}, FP2={FP2}, FN={FN}")
            
            # Calculate metrics
            total = TP + TN + FP1 + FP2 + FN
            accuracy = (TP + TN) / total if total > 0 else 0
            precision = TP / (TP + FP1 + FP2) if (TP + FP1 + FP2) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}\n")
        
        # Save model checkpoint
        if (epoch + 1) % save_freq == 0:
            model_save_path = os.path.join(work_dir, f"model_epoch_{epoch + 1}.pth")
            print(f"Saving model checkpoint to {model_save_path}...")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': experiment_config,
            }, model_save_path)
            print(f"✓ Model checkpoint saved\n")
    
    # Save final model
    final_model_path = os.path.join(work_dir, "model_final.pth")
    print(f"\nSaving final model to {final_model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': experiment_config,
    }, final_model_path)
    print(f"✓ Final model saved")
    
    # Clean up checkpoint file
    checkpoint_path = os.path.join(work_dir, 'training_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"✓ Checkpoint file cleaned up")
    
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TrackNetV4 model with PyTorch"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['TrackNetV4_TypeA', 'TrackNetV4_TypeB'],
        help="Name of the model to use"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['tennis_game_level_split', 'tennis_clip_level_split', 'badminton', 'new_tennis'],
        help="Name of the dataset to use"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing sequences within each .npy file")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--height", type=int, default=288, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--tol", type=int, default=4, help="Tolerance for evaluation (pixels)")
    parser.add_argument("--work_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--save_freq", type=int, default=1, help="Save model every N epochs")
    parser.add_argument("--disable_eval_on_train", action="store_true", help="Disable evaluation on training set")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    
    args = parser.parse_args()
    main(args)