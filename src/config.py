import time

# Data paths
train_csv_file = 'data/skin_cancer.v2i.multiclass/train/processed_classes.csv'
train_root_dir = 'data/skin_cancer.v2i.multiclass/train'
val_csv_file = 'data/skin_cancer.v2i.multiclass/valid/processed_classes.csv'
val_root_dir = 'data/skin_cancer.v2i.multiclass/valid'

# Checkpoint directory
checkpoint_dir = 'data/checkpoints/'

# Current timestamp for saving models and checkpoints
current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

# Model and training parameters
num_classes = 8
learning_rate = 0.001
epochs = 10 # Default value, can be overridden by command line arguments
batch_size = 64 # Default value, can be overridden by command line arguments

# Wandb settings
wandb_project = "skin-cancer-classification" 