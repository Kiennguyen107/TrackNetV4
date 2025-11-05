import os
import json
import numpy as np
import cv2
import pandas as pd
from glob import glob
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import concurrent.futures

from util import genHeatMap
from constants import (
    SIGMA,
    MAG,
    TENNIS_DATASET_CLIP_PLAYER_CSV_DIR,
    TENNIS_DATASET_SELECT_CLIP,
    WIDTH,
    HEIGHT,
    TENNIS_DATASET_GAME_LEVEL_SPLIT_CSV,
    TENNIS_DATASET_CLIP_LEVEL_SPLIT_CSV,
)


class BaseDataset(Dataset):
    def __init__(self, root_dir, mode, split_type='game_level', target_img_height=HEIGHT, target_img_width=WIDTH, sequence_dim=(3,3), mag=MAG, sigma=SIGMA, shuffle=True):
        self.root_dir = root_dir
        self.mode = mode
        self.target_img_height = target_img_height
        self.target_img_width = target_img_width
        self.sequence_dim = sequence_dim
        # self.processed_folder = os.path.join(root_dir, "processed_data", mode)
        self.split_type = split_type
        self.processed_folder = os.path.join(root_dir, "processed_data", self.split_type, self.mode)
        self.shuffle = shuffle
        self.mag = mag
        self.sigma = sigma
        self.transform = ToTensor()
        self.resize = Resize((self.target_img_height, self.target_img_width))
    
    def __len__(self):
        """
        Returns the number of processed sample pairs (x and y).
        """
        if not self._is_processed():
            return 0
        self.file_list = glob(os.path.join(self.processed_folder, "x_data_*.npy"))
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Loads x and y data for the given index.
        """
        if not self._is_processed():
            return None, None
        self.file_list = glob(os.path.join(self.processed_folder, "x_data_*.npy"))
        x_path = self.file_list[idx]
        y_path = x_path.replace("x_data", "y_data")
        
        x_data = np.load(x_path)
        y_data = np.load(y_path)

        x_tensor = torch.from_numpy(x_data).float()
        y_tensor = torch.from_numpy(y_data).float()

        return x_tensor, y_tensor

    def _is_processed(self):
        """
        Checks if the processed .npy files exist for the current mode.
        """
        if not os.path.exists(self.processed_folder):
            print(f"Processed folder does not exist: {self.processed_folder}")
            return False

        npy_files = glob(os.path.join(self.processed_folder, "*.npy"))
        if npy_files:
            return True
        else:
            print(f"No .npy files found in {self.processed_folder}")
            return False

    def process_data(self):
        """
        Processes the dataset and creates processed .npy files.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the process_data method.")

class TennisDataset(BaseDataset):
    def __init__(self, root_dir, mode, split_type='game_level', **kwargs):
        self.split_type = split_type
        self.mode = mode
        self.valid_clips = []
        self.processed_folder = os.path.join(root_dir, "processed_data", self.split_type, self.mode)
        if mode == 'val': # The split files use 'test' for validation set
            mode = 'test'
        super().__init__(root_dir, mode, **kwargs)

    def process_data(self):
        # read the select clip dirs
        with open(TENNIS_DATASET_SELECT_CLIP, 'r') as f:
            selected_clips = f.read().splitlines()
        self.valid_clips = []
        for i in selected_clips:
            s=tuple(i.strip().split('/'))
            self.valid_clips.append(s)
        print(self.valid_clips)
        
        if self.split_type == "game_level":
            self._process_game_level()
        elif self.split_type == "clip_level":
            self._process_clip_level()
        print("Processing completed.")


    @staticmethod
    def get_clip_level_split(csv_path_or_df):
        """
        Given a CSV path or DataFrame with columns 'game', 'clip', and 'set',
        returns two dictionaries: train_dict and test_dict.
        Each dictionary maps a game to a list of its associated clips.
        """
        df = pd.read_csv(csv_path_or_df) if isinstance(csv_path_or_df, str) else csv_path_or_df

        train_dict = defaultdict(list)
        test_dict = defaultdict(list)

        for _, row in df.iterrows():
            if row['set'] == 'train':
                train_dict[row['game']].append(row['clip'])
            elif row['set'] == 'test':
                test_dict[row['game']].append(row['clip'])

        return dict(train_dict), dict(test_dict)

    @staticmethod
    def get_game_level_split(csv_path):
        """
        Reads a CSV with columns 'game' and 'set', and returns two lists:
        train_games and test_games.
        """
        df = pd.read_csv(csv_path)
        train_games = df[df['set'] == 'train']['game'].tolist()
        test_games = df[df['set'] == 'test']['game'].tolist()
        return train_games, test_games
    
    # def _process_clip_level(self):
    #     df = pd.read_csv(TENNIS_DATASET_CLIP_LEVEL_SPLIT_CSV)
    #     set_data = df[df['set'] == self.mode]
        
    #     os.makedirs(self.processed_folder, exist_ok=True)
    #     count = 1
        
    #     for _, row in set_data.iterrows():
    #         game = row['game']
    #         clip = row['clip']
    #         game_folder = os.path.join(self.root_dir, "Dataset", game)
    #         clip_folder = os.path.join(game_folder, clip)
    #         print(f"Processing game: {game}, clip: {clip}")
    #         x_data, y_data = self._process_clip(clip_folder)
    #         if x_data is not None:
    #             for i in range(x_data.shape[0]):
    #                 np.save(os.path.join(self.processed_folder, f'x_data_{count}.npy'), x_data[i])
    #                 np.save(os.path.join(self.processed_folder, f'y_data_{count}.npy'), y_data[i])
    #                 count += 1

    # def _process_game_level(self):
    #     df = pd.read_csv(TENNIS_DATASET_GAME_LEVEL_SPLIT_CSV)
    #     set_data = df[df['set'] == self.mode]
        
    #     os.makedirs(self.processed_folder, exist_ok=True)
    #     count = 1

    #     for _, row in set_data.iterrows():
    #         game = row['game']
    #         game_folder = os.path.join(self.root_dir, "Dataset", game)
    #         clips = [d for d in os.listdir(game_folder) if os.path.isdir(os.path.join(game_folder, d))]
    #         for clip in clips:
    #             clip_folder = os.path.join(game_folder, clip)
    #             print(f"Processing game: {game}, clip: {clip}")
    #             x_data, y_data = self._process_clip(clip_folder)
    #             if x_data is not None:
    #                 for i in range(x_data.shape[0]):
    #                     np.save(os.path.join(self.processed_folder, f'x_data_{count}.npy'), x_data[i])
    #                     np.save(os.path.join(self.processed_folder, f'y_data_{count}.npy'), y_data[i])
    #                     count += 1
        

    def _process_clip_level(self):
        train_set, test_set = self.get_clip_level_split(TENNIS_DATASET_CLIP_LEVEL_SPLIT_CSV)
        self._process_set(train_set, os.path.join(self.root_dir, "processed_data", "clip_level", "train"))
        self._process_set(test_set, os.path.join(self.root_dir, "processed_data", "clip_level", "test"))

    def _process_game_level(self):
        train_games, test_games = self.get_game_level_split(TENNIS_DATASET_GAME_LEVEL_SPLIT_CSV)
        self._process_set(train_games, os.path.join(self.root_dir, "processed_data", "game_level", "train"), use_clip_list=False)
        self._process_set(test_games, os.path.join(self.root_dir, "processed_data", "game_level", "test"), use_clip_list=False)

    def _process_set(self, set_data, save_data_dir, use_clip_list=True):
        """
        Process a set of clips or games.
        
        :param set_data: Either a dictionary (game->list of clips) or a list of game names.
        :param save_data_dir: Directory to save processed data.
        :param use_clip_list: If True, set_data is a dict of game:clips; if False, process all clips in each game folder.
        """
        os.makedirs(save_data_dir, exist_ok=True)
        count = 1


        def process_clip_wrapper(args):
                game, clip, game_folder, save_data_dir, count = args
                clip_folder = os.path.join(game_folder, clip)
                print(f"Processing game: {game}, clip: {clip}", flush=True)
                x_data, y_data = self._process_clip(game, clip, clip_folder)
                self._save_data(save_data_dir, count, x_data, y_data)
                return count


        if use_clip_list:
            # set_data is a dict: game -> list of clips
            seen = set()
            tasks = []
            for game, clips in set_data.items():
                game_folder = os.path.join(self.root_dir, "Dataset", game)
                for clip in clips:
                    key = (game, clip)
                    if key not in self.valid_clips: 
                        continue
                    if key not in seen:
                        tasks.append((game, clip, game_folder, save_data_dir, count))
                        seen.add(key)
                        count += 1
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(executor.map(process_clip_wrapper, tasks))
        else:
            # set_data is a list of games; process all clips in each game folder.
            seen = set()
            tasks = []
            for game in set_data:
                game_folder = os.path.join(self.root_dir, "Dataset", game)
                clips = [d for d in os.listdir(game_folder) if os.path.isdir(os.path.join(game_folder, d))]
                for clip in clips:
                    key = (game, clip)
                    if key not in self.valid_clips: 
                        continue
                    if key not in seen:
                        tasks.append((game, clip, game_folder, save_data_dir, count))
                        seen.add(key)
                        count += 1
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(executor.map(process_clip_wrapper, tasks))
    
    def _save_data(self, save_dir, count, x_data, y_data):
        """
        Saves processed data as npy files.
        """
        np.save(os.path.join(save_dir, f'x_data_{count}.npy'), x_data)
        np.save(os.path.join(save_dir, f'y_data_{count}.npy'), y_data)

    def _process_clip(self, game, clip, clip_folder):
        label_path = os.path.join(clip_folder, 'Label.csv')
        player_label_path = os.path.join(TENNIS_DATASET_CLIP_PLAYER_CSV_DIR, f'{game}_{clip}_players.csv')
        if not os.path.exists(label_path):
            return None, None
        if not os.path.exists(player_label_path):
            return None, None
        label_data = pd.read_csv(label_path)
        file_names = label_data['file name'].values
        visibility = label_data['visibility'].values
        x_coords = label_data['x-coordinate'].values
        y_coords = label_data['y-coordinate'].values

        player_data = pd.read_csv(player_label_path)
        frames = player_data['frame'].values
        player_ids = player_data['player_id'].values
        x1_coords = player_data['x1'].values
        y1_coords = player_data['y1'].values
        x2_coords = player_data['x2'].values
        y2_coords = player_data['y2'].values
        

        num_frames = file_names.shape[0]
        
        if num_frames == 0:
            return None, None

        sample_image_path = os.path.join(clip_folder, file_names[0])
        if not os.path.exists(sample_image_path):
            return None, None
            
        sample_image = np.array(Image.open(sample_image_path))
        ratio = sample_image.shape[0] / self.target_img_height

        x_data_list = []
        y_data_list = []

        for i in range(num_frames - (self.sequence_dim[0] - 1)):
            frames_sequence = []
            for j in range(self.sequence_dim[0]):
                frame_path = os.path.join(clip_folder, str(file_names[i + j]))
                if not os.path.exists(frame_path):
                    continue

                img = Image.open(frame_path)
                img = self.resize(img)
                img_tensor = self.transform(img)
                frames_sequence.append(img_tensor)

            if len(frames_sequence) != self.sequence_dim[0]:
                continue

            # Stack frames and then flatten the first two dimensions
            x_tensor = torch.stack(frames_sequence, dim=0) # Shape: (3, 3, H, W)
            x_tensor = x_tensor.view(-1, self.target_img_height, self.target_img_width) # Shape: (9, H, W)
            x_data_list.append(x_tensor.numpy())

            # Ball heatmap sequence (existing)
            heatmap_sequence = []
            for j in range(self.sequence_dim[1]):
                if visibility[i + j] == 0:
                    heatmap = genHeatMap(self.target_img_width, self.target_img_height, -1, -1, self.sigma, self.mag)
                else:
                    heatmap = genHeatMap(
                        self.target_img_width, self.target_img_height,
                        int(x_coords[i + j] / ratio),
                        int(y_coords[i + j] / ratio),
                        self.sigma, self.mag
                    )
                heatmap_sequence.append(heatmap)
            heatmap_sequence = np.stack(heatmap_sequence, axis=0)  # Shape: (seq_len, H, W)

            # Player heatmap sequence
            player_heatmap_sequence = []
            for j in range(self.sequence_dim[1]):
                frame_idx = i + j
                frame_num = frames[frame_idx] if frame_idx < len(frames) else None
                # Find all players in this frame
                frame_players = player_data[player_data['frame'] == frame_num]
                player_heatmap = np.zeros((self.target_img_height, self.target_img_width), dtype=np.float32)
                for _, prow in frame_players.iterrows():
                    # Scale coordinates
                    x1 = int(prow['x1'] / ratio)
                    y1 = int(prow['y1'] / ratio)
                    x2 = int(prow['x2'] / ratio)
                    y2 = int(prow['y2'] / ratio)
                    # Draw filled rectangle for player region
                    cv2.rectangle(player_heatmap, (x1, y1), (x2, y2), 1.0, thickness=-1)
                player_heatmap_sequence.append(player_heatmap)
            player_heatmap_sequence = np.stack(player_heatmap_sequence, axis=0)  # Shape: (seq_len, H, W)

            y_data_list.append(np.concatenate([
                heatmap_sequence[:, None, :, :],            # (seq_len, 1, H, W)
                player_heatmap_sequence[:, None, :, :]      # (seq_len, 1, H, W)
            ], axis=1))  # (seq_len, 2, H, W)

        if not x_data_list:
            return None, None

        x_data = np.asarray(x_data_list, dtype='float32')
        y_data = np.asarray(y_data_list, dtype='float32')

        return x_data, y_data
