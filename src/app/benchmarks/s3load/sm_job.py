import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime

def submit_boto3_benchmark(num_workers=4, wait=False):
    """
    Submit a SageMaker training job for boto3 S3 loading benchmark.
    
    Args:
        num_workers: Number of DataLoader workers
        wait: Whether to wait for job completion
    """
    # Get SageMaker session and role
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Create unique job name
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    job_name = f'boto3-benchmark-{timestamp}'
    
    # Configure PyTorch estimator
    estimator = PyTorch(
        entry_point='boto3_benchmark.py',
        source_dir='benchmarks/training_scripts',
        role=role,
        instance_count=1,
        instance_type='ml.g4dn.xlarge',
        framework_version='2.2.0',  # PyTorch version
        py_version='py311',
        hyperparameters={
            'epochs': 3,
            'batch-size': 32,
            'learning-rate': 0.001,
            'num-workers': num_workers,
            'bucket': 'lab-genai-gal12012026',
            'prefix': 'benchmarks/s3loading/data',
            'num-images': 1000,
            'num-classes': 10,
        },
        disable_profiler=True,  # We're doing our own metrics
        debugger_hook_config=False,
    )
    
    print(f"Submitting job: {job_name}")
    print(f"Instance type: ml.g4dn.xlarge")
    print(f"Num workers: {num_workers}")
    
    # Submit the training job
    estimator.fit(wait=wait, job_name=job_name)
    
    if wait:
        print(f"\nJob completed!")
    else:
        print(f"\nJob submitted!")
    
    return estimator

if __name__ == "__main__":
    # Submit and wait for completion
    submit_boto3_benchmark(num_workers=1, wait=True)