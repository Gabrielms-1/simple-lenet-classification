import sagemaker
from sagemaker.pytorch import PyTorch
import configparser
from datetime import datetime

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

s3_prefix = "fashionmnist"

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
        "input_dataset_dir": training_config['S3']['input_dataset_dir'],
        "num_classes": int(training_config['TRAINING']['num_classes']),
        "resize": int(training_config['TRAINING']['resize']),
    },
    base_job_name=f"cad-fashionmnist-classification-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
     tags=[
        {"Key": "Application", "Value": network_config['TAGS']['application']},
        {"Key": "Cost Center", "Value": network_config['TAGS']['cost_center']}
    ],
    subnets=network_config['NETWORK']['subnets'].split(','),
    security_group_ids=network_config['NETWORK']['security_group_ids'].split(','),
    checkpoint_s3_uri=training_config['S3']['checkpoint_dir'],
    checkpoint_local_path="/opt/ml/checkpoints",
    output_s3_uri=training_config['S3']['output_dir']+"/testing2",
    output_path=training_config['S3']['output_dir']+"/testing1",
    environment={
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    }
)

estimator.fit(
    inputs={
        "train": training_dir,
        "val": val_dir,
    }
)