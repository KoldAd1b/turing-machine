# Data Directory

This project expects the ADE20K/ADEChallengeData2016 semantic segmentation dataset.

Do not commit the dataset files to this repository. Place or symlink the dataset here
when running locally, or pass the dataset location with `--path_to_data`.

Expected layout:

```text
data/
└── ADEChallengeData2016/
    ├── images/
    │   ├── training/
    │   │   └── ADE_train_00000001.jpg
    │   └── validation/
    │       └── ADE_val_00000001.jpg
    └── annotations/
        ├── training/
        │   └── ADE_train_00000001.png
        └── validation/
            └── ADE_val_00000001.png
```

Example:

```bash
python unet_train_ade20k.py \
  --path_to_data data/ADEChallengeData2016 \
  --image_size 256
```
