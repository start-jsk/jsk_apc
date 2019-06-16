## Usage

```bash
# download dataset and generate heightmap
./get_heightmaps.py
./check_heightmaps.py

# view dataset
./view_pinch_dataset.py
./view_suction_dataset.py

# train model
./train.py pinch --model rgb+depth
./train.py suction --model rgb+depth

# check result of training
./summarize_logs.py
./plot_pinch_resolution.py
```
