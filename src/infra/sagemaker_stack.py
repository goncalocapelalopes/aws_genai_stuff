from aws_cdk import Stack
from aws_cdk import aws_sagemaker as sagemaker
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_ec2 as ec2
from constructs import Construct

class SageMakerStack(Stack):   
    def __init__(self, scope: Construct, construct_id: str, data_bucket: s3.IBucket, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # IAM Role for SageMaker
        self.sagemaker_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
            ]
        )
        
        # Grant read/write permissions to the data bucket
        data_bucket.grant_read_write(self.sagemaker_role)
        
        # Look up the default VPC
        vpc = ec2.Vpc.from_lookup(
            self,
            "DefaultVPC",
            is_default=True
        )

        # Get subnet IDs from the VPC
        subnet_ids = [subnet.subnet_id for subnet in vpc.private_subnets]
        # If no private subnets, use public subnets
        if not subnet_ids:
            subnet_ids = [subnet.subnet_id for subnet in vpc.public_subnets]

        # SageMaker Domain
        self.domain = sagemaker.CfnDomain(
            self,
            "SageMakerDomain",
            domain_name="Labyrinth",
            auth_mode="IAM",
            vpc_id=vpc.vpc_id,
            subnet_ids=subnet_ids,
            app_network_access_type="PublicInternetOnly",
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=self.sagemaker_role.role_arn
            )
        )
        
        #User Profile
        self.user_profile = sagemaker.CfnUserProfile(
            self,
            "SageMakerUserProfile",
            domain_id=self.domain.attr_domain_id,
            user_profile_name="gon",
            user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                execution_role=self.sagemaker_role.role_arn
            )
        )
