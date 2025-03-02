import sagemaker
from sagemaker.pytorch import PyTorch
import configparser
from datetime import datetime
from zoneinfo import ZoneInfo

network_config = configparser.ConfigParser()
network_config.read('sagemaker/credentials.ini')  

training_config = configparser.ConfigParser()
training_config.read('sagemaker/config.ini')


sagemaker_session_bucket = training_config['S3']['bucket']
#sagemaker_session = sagemaker.Session()

session = sagemaker.Session(
    default_bucket=sagemaker_session_bucket,
)

bucket = session.default_bucket()
role = sagemaker.get_execution_role()

training_dir = (training_config['S3']['train_dir'])
val_dir = (training_config['S3']['val_dir'])

s3_prefix = "alzheimer-brain"

timestamp = datetime.now(ZoneInfo('America/Sao_Paulo')).strftime('%Y%m%d_%H-%M-%S')

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role=role,
    pytorch_version="2.2",
    framework_version="2.2",
    py_version="py310",
    hyperparameters={
        "epochs": int(training_config['TRAINING']['epochs']),
        "batch_size": int(training_config['TRAINING']['batch_size']),
        "learning_rate": float(training_config['TRAINING']['learning_rate']),
        "num_classes": int(training_config['TRAINING']['num_classes']),
        "resize": int(training_config['TRAINING']['resize']),
        "wandb_mode": "offline",
    },
    base_job_name=f"cad-alzheimer-brain-classification-RGB",
    tags=[
        {"Key": "Application", "Value": network_config['TAGS']['application']},
        {"Key": "Cost Center", "Value": network_config['TAGS']['cost_center']}
    ],
    subnets=network_config['NETWORK']['subnets'].split(','),
    security_group_ids=network_config['NETWORK']['security_group_ids'].split(','),
    checkpoint_s3_uri=training_config['S3']['checkpoint_dir'] + "/" + timestamp,
    checkpoint_local_path="/opt/ml/checkpoints",
    output_path=training_config['S3']['output_dir'] + "/" + timestamp,
    environment={
        "WANDB_API_KEY": network_config['WANDB']['wandb_api_key'],
        "WANDB_MODE": "offline",
        "WANDB_DIR": "/opt/ml/model/",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    },
    metric_definitions=[
        {'Name': 'train:loss', 'Regex': 'Loss: ([0-9\\.]+)'},
        {'Name': 'train:accuracy', 'Regex': 'Accuracy: ([0-9\\.]+)'},
        {'Name': 'val:loss', 'Regex': 'Validation Loss: ([0-9\\.]+)'},
        {'Name': 'val:accuracy', 'Regex': 'Validation Accuracy: ([0-9\\.]+)'},
        {'Name': 'val:recall', 'Regex': 'Validation Recall: ([0-9\\.]+)'},
        {'Name': 'val:f1', 'Regex': 'Validation F1: ([0-9\\.]+)'},
    ],
    enable_sagemaker_metrics=True
)

estimator.fit(
    inputs={
        "train": training_dir,
        "val": val_dir,
    }
)