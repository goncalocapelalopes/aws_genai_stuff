import boto3
from torch.utils.data import Dataset
from PIL import Image
import io

class S3ImageDataset(Dataset):
    def __init__(self, bucket, prefix, num_images=1000, num_classes=10, transform=None):
        self.bucket = bucket
        self.prefix = prefix
        self.s3_client = boto3.client('s3')
        self.transform = transform
        
        # List all images in S3
        self.image_keys = [] 
    
        for i in range(num_images):
            class_id = i % num_classes
            filename = f"{self.prefix}/img_{i:04d}_class_{class_id}.jpg"
            self.image_keys.append(filename)

    def _extract_label_from_key(self, key):
        return int(key.split('_class_')[1].split('.')[0])

    def __len__(self):
        return len(self.image_keys)
    
    def __getitem__(self, idx):
        key = self.image_keys[idx]
        
        # Download from S3
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        except Exception as e:
            print(f"Error loading {key}: {e}")
            raise
        image_bytes = response['Body'].read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms (to tensor, normalize)
        if self.transform:
            image = self.transform(image)
        
        # Extract label
        label = self._extract_label_from_key(key)
        
        return image, label