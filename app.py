import os
from aws_cdk import App, Environment

from src.infra.sagemaker_stack import SageMakerStack
from src.infra.storage_stack import StorageStack

app = App()

env = Environment(
    account=os.environ["CDK_DEFAULT_ACCOUNT"],
    region=os.environ.get("CDK_DEFAULT_REGION", "eu-west-1")
)

sagemaker_stack = SageMakerStack(app, "SageMakerStack", env=env)
storage_stack = StorageStack(app, "StorageStack", env=env)

app.synth()