from aws_cdk import Stack, RemovalPolicy
from aws_cdk import aws_s3 as s3
from constructs import Construct

class StorageStack(Stack):   
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 Bucket for storing data
        self.data_bucket = s3.Bucket(
            self,
            "LabBucket",
            bucket_name="lab-genai-gal12012026",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            versioned=False,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL
        )