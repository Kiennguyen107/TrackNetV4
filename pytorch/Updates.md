# Dataset changes

- Added a new dataset to involve player detection and tracking.

- Scripts to generate player bounding boxes and CSV files for player positions have been added. 

Run the following script to generate player data in parallel:

```bash
bash multi_thread_gen_pyr_data.sh
```
The main program is the `gen_player_data.py` script, which uses a pre-trained YOLO model to detect players in video frames.

Make sure the deps are fullfilled