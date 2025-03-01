import sagemaker
from sagemaker.pytorch import PyTorch
import configparser
from datetime import datetime

network_config = configparser.ConfigParser()
network_config.read('sagemaker/credentials.ini')  

training_config = configparser.ConfigParser()
training_config.read('sagemaker/config.ini')


sagemaker_session_bucket = training_config['S3']['bucket']

role = sagemaker.get_execution_role()

training_dir = "file://data/plant_leaves_disease/processed/train/"
val_dir = "file://data/plant_leaves_disease/processed/valid/"

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    instance_type="local",
    instance_count=1,
    role=role,
    framework_version="2.2",
    pytorch_version="2.2",
    py_version="py310",
    hyperparameters={
        "epochs": int(training_config['TRAINING']['epochs']),
        "batch_size": int(training_config['TRAINING']['batch_size']),
        "learning_rate": float(training_config['TRAINING']['learning_rate']),
        "num_classes": int(training_config['TRAINING']['num_classes']),
        "resize": int(training_config['TRAINING']['resize']),
        "wandb_mode": "offline"
    },
    requirements_file="requirements.txt",
    base_job_name=f"cad-plant-disease-classification-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    output_path="file://data/plant_leaves_disease/models/",
    environment={
        "WANDB_API_KEY": network_config['WANDB']['wandb_api_key'],
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    },
    metric_definitions=[
            {
                "Name": "train_loss",
                "Regex": "train_loss: ([0-9.]+)"
            },  
            {
                "Name": "val_loss",
                "Regex": "val_loss: ([0-9.]+)"
            },
            {
                "Name": "train_acc",
                "Regex": "train_acc: ([0-9.]+)"
            },
            {
                "Name": "val_acc",
                "Regex": "val_acc: ([0-9.]+)"
            },
            {
                "Name": "val_recall",
                "Regex": "val_recall: ([0-9.]+)"
            }
        ]
)

estimator.fit(
    inputs={
        "train": training_dir,
        "val": val_dir,
    }
)