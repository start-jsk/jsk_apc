# `grasp_data_generator`: Automatic dataset generation for dual-arm grasping 

## Preparing data

### Downloading original object data to `../data/compressed`

```bash
python download_object_data.py
```

### Downloading CNN model for background subtraction to `../data/models`

```bash
python download_models.py
```

### Copying object data to `../data/objects`

```
python copy_object_data.py
```

## Generating initial datasets in `../data/training_data` (no sampling)

### Human annotation

```
python generate_mask.py
python annotate_grasping_point.py
python generate_training_data.py -n 2000
```

### No human annotation

```
python generate_mask.py -g
python generate_training_data.py -n 2000 --no-human
```

### No human annotation & instance mask

```
python generate_mask.py -g
python generate_training_data.py -n 2000 --no-human --instance
```

## Generating a finetuning dataset with `../data/sampling_data`

### Generating object mask for `../data/sampling_data`

## Use pre-made datasets

### Downloading pre-made datasets to `../data/compressed`

```bash
python download_datasets.py
```

### Extracting pre-made datasets to `../data/training_data` and `../data/finetuning_data`

```bash
python extract_datasets.py
```
