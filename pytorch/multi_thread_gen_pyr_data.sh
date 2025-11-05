max_jobs=6
pids=()

for game in {1..10}; do
    for clip_dir in data/tennis/Dataset/game${game}/Clip*/; do
        output_file="outputs/game${game}_$(basename "$clip_dir")_players.mp4"
        csv_file="csv/game${game}_$(basename "$clip_dir")_players.csv"
        (python gen_player_data.py "$clip_dir" --output_path "$output_file" --csv_path "$csv_file" --fps 24 --calibration_frames 60) &
        pids+=($!)
        # If max_jobs reached, wait for any to finish
        if (( ${#pids[@]} >= max_jobs )); then
            wait -n
            # Remove finished pids
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[i]}" 2>/dev/null; then
                    unset 'pids[i]'
                fi
            done
        fi
    done
done

# Wait for remaining processes
wait