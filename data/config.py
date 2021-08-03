# config.py
import os.path


# centernet config
train_cfg = {
    'train_size': (1280, 720),
    'val_size': (1280, 720),
    'lr_epoch': (100, 150),
    'max_epoch': 200
}

train_baseline_cfg = {
    'train_size': (1280, 720),
    'val_size': (1280, 720),
    'lr_epoch': (90, 120),
    'max_epoch': 150
}
