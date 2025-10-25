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
                     Allowed values: Baseline_TrackNetV2, TrackNetV4_TypeA, TrackNetV4_TypeB.
    --dataset      : Name of the dataset to use.
                     Allowed values: tennis_game_level_split, tennis_clip_level_split, badminton, new_tennis.
    --batch_size   : Batch size for training (default: 3).
    --learning_rate: Learning rate for the optimizer (default: 1.0).
    --height       : Target image height (default: 288).
    --width        : Target image width (default: 512).
    --epochs       : Number of epochs for training (default: 30).
    --tol          : Tolerance for the outcome evaluation (default: 4).
    --model_path   : Path to a pretrained model (.keras) to load before training (optional).
    --work_dir     : Directory to save the trained models (default: "./models").
    --save_freq    : Frequency (in epochs) to save model checkpoints (default: 1).

Note:
    If the default work directory is used, a timestamp will be appended to create a unique directory.
"""

#!/usr/bin/env python
"""
Training Script with Checkpoint Support
-------------------------

This script trains a TrackNet model with:
- GPU memory optimization
- Checkpoint system for resuming interrupted training
- Automatic model saving

Usage:
    python src/train.py --model_name <MODEL> --dataset <DATASET> [options]

Example:
    python src/train.py --model_name Baseline_TrackNetV2 --dataset tennis_clip_level_split \
        --batch_size 2 --learning_rate 0.001 --epochs 30 --resume
"""

import argparse
import datetime
import os
import gc
import json

# CRITICAL: Configure GPU memory BEFORE importing tensorflow
import tensorflow as tf

# Enable GPU memory growth to prevent OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint

from util import custom_loss, get_dataset, get_model, outcome
from models.TrackNetV4 import (
    MotionPromptLayer,
    FusionLayerTypeA,
    FusionLayerTypeB
)


def save_checkpoint(work_dir, epoch, batch_idx, experiment_config):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'config': experiment_config,
        'timestamp': str(datetime.datetime.now())
    }
    checkpoint_path = os.path.join(work_dir, 'training_checkpoint.json')
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"💾 Checkpoint saved: Epoch {epoch}, Batch {batch_idx}")


def load_checkpoint(work_dir):
    """Load training checkpoint if exists"""
    checkpoint_path = os.path.join(work_dir, 'training_checkpoint.json')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint
    return None


def find_latest_model(work_dir):
    """Find the latest saved model in work_dir"""
    import glob
    model_files = glob.glob(os.path.join(work_dir, 'model_epoch_*.keras'))
    if not model_files:
        return None
    
    # Extract epoch numbers and find max
    epochs = []
    for f in model_files:
        try:
            epoch_num = int(f.split('model_epoch_')[-1].replace('.keras', ''))
            epochs.append((epoch_num, f))
        except:
            continue
    
    if epochs:
        latest = max(epochs, key=lambda x: x[0])
        return latest[1], latest[0]
    return None


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
    resume = args.resume

    # If using default work directory, append a timestamp for uniqueness
    if work_dir == "./models" and not resume:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(work_dir, timestamp)

    # Create the work directory if it doesn't exist
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
        "model_path": model_path,
        "work_dir": work_dir,
        "save_freq": save_freq,
    }

    # Try to resume from checkpoint
    start_epoch = 0
    start_batch = 0
    
    if resume:
        checkpoint = load_checkpoint(work_dir)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch_idx']
            print(f"\n📌 RESUMING FROM CHECKPOINT")
            print(f"   Work directory: {work_dir}")
            print(f"   Last epoch: {start_epoch}")
            print(f"   Last batch: {start_batch}")
            print(f"   Timestamp: {checkpoint['timestamp']}\n")
            
            # Find and load latest model
            latest_model = find_latest_model(work_dir)
            if latest_model:
                model_path = latest_model[0]
                print(f"✓ Loading model from: {model_path}")
            else:
                print("⚠️  No saved model found, starting fresh")
                start_epoch = 0
                start_batch = 0
        else:
            print(f"\n⚠️  No checkpoint found in {work_dir}")
            print("Starting fresh training...\n")

    # Print configuration
    print("\n" + "="*60)
    print("Training Configurations:")
    for key, value in experiment_config.items():
        print(f"  {key}: {value}")
    print(f"  resume: {resume}")
    print(f"  start_epoch: {start_epoch + 1}/{epochs}")
    print("="*60 + "\n")

    # Load or create model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model = load_model(
            model_path, 
            custom_objects={
                'MotionPromptLayer': MotionPromptLayer,
                'FusionLayerTypeA': FusionLayerTypeA,
                'FusionLayerTypeB': FusionLayerTypeB,
                'custom_loss': custom_loss
            }
        )
        print(f"✓ Model loaded from checkpoint\n")
    else:
        print("Creating new model...")
        model = get_model(model_name, height, width)
        print(f"✓ Model created: {model_name}\n")

    # Load training dataset
    print("Loading training dataset...")
    dataset_train = get_dataset(dataset_name, "train", height, width)
    print(f"✓ Dataset loaded: {len(dataset_train)} samples\n")

    # Compile the model
    print("Compiling model...")
    model.compile(
        loss=custom_loss,
        optimizer=Adadelta(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    print("✓ Model compiled\n")

    # Main training loop
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        # Train on each batch in the training dataset
        batch_count = 0
        for x_train, y_train in dataset_train:
            batch_count += 1
            
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and batch_count <= start_batch:
                print(f"⏭️  Skipping batch {batch_count} (already processed)")
                del x_train, y_train
                gc.collect()
                continue
            
            print(f"Training on batch {batch_count}/{len(dataset_train)}...", flush=True)
            
            try:
                model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
                
                # Save checkpoint after each batch
                save_checkpoint(work_dir, epoch, batch_count, experiment_config)
                
            except KeyboardInterrupt:
                print(f"\n\n⚠️  TRAINING INTERRUPTED BY USER")
                print(f"✓ Progress saved at Epoch {epoch + 1}, Batch {batch_count}")
                print(f"▶️  To resume, run with --resume --work_dir {work_dir}\n")
                return
            
            except Exception as e:
                print(f"❌ ERROR in batch {batch_count}: {e}")
                import traceback
                traceback.print_exc()
                print(f"⏩ Continuing to next batch...\n")
            
            finally:
                # CRITICAL: Clear memory after each batch
                del x_train, y_train
                gc.collect()
                if hasattr(tf.keras.backend, 'clear_session'):
                    tf.keras.backend.clear_session()

        # Reset start_batch for next epoch
        start_batch = 0

        # Evaluate model performance on the training set (if enabled)
        if not args.disable_eval_on_train:
            print(f"\nEvaluating on training set...")
            TP = TN = FP1 = FP2 = FN = 0
            
            eval_count = 0
            for x_train, y_train in dataset_train:
                eval_count += 1
                if eval_count % 10 == 0:
                    print(f"  Evaluated {eval_count}/{len(dataset_train)} batches...", flush=True)
                
                try:
                    y_pred = model.predict(x_train, batch_size=batch_size, verbose=0)
                    y_pred = (y_pred > 0.5).astype('float32')

                    tp, tn, fp1, fp2, fn = outcome(y_pred, y_train, tol)
                    TP += tp
                    TN += tn
                    FP1 += fp1
                    FP2 += fp2
                    FN += fn
                except Exception as e:
                    print(f"⚠️  Eval error on batch {eval_count}: {e}")

                # CRITICAL: Clear memory
                del x_train, y_train
                if 'y_pred' in locals():
                    del y_pred
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

        # Save model checkpoint based on frequency
        if (epoch + 1) % save_freq == 0:
            model_save_path = os.path.join(work_dir, f"model_epoch_{epoch + 1}.keras")
            print(f"Saving model checkpoint to {model_save_path}...")
            model.save(model_save_path)
            print(f"✓ Model checkpoint saved\n")

    # Save the final model after training
    final_model_path = os.path.join(work_dir, "model_final.keras")
    print(f"\nSaving final model to {final_model_path}...")
    model.save(final_model_path)
    print(f"✓ Final model saved")
    
    # Clean up checkpoint file
    checkpoint_path = os.path.join(work_dir, 'training_checkpoint.json')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"✓ Checkpoint file cleaned up")
    
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TrackNet model with checkpoint support."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['Baseline_TrackNetV2', 'TrackNetV4_TypeA', 'TrackNetV4_TypeB'],
        help="Name of the model to use."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['tennis_game_level_split', 'tennis_clip_level_split', 'badminton', 'new_tennis'],
        help="Name of the dataset to use."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--height", type=int, default=288, help="Target height of images")
    parser.add_argument("--width", type=int, default=512, help="Target width of images")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--tol", type=int, default=4, help="Tolerance for outcome evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model (.keras) to load before training"
    )
    parser.add_argument(
        "--work_dir", 
        type=str, 
        default="./models", 
        help="Directory to save models and checkpoints"
    )
    parser.add_argument(
        "--save_freq", 
        type=int, 
        default=1, 
        help="Save model every N epochs"
    )
    parser.add_argument(
        "--disable_eval_on_train",
        action="store_true",
        help="Disable evaluation on training set after each epoch"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint in work_dir"
    )
    
    args = parser.parse_args()
    main(args)