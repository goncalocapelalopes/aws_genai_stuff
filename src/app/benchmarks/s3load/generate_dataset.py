import io
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3

def generate_random_image() -> Image.Image:
    """
    Generate a random RGB image of size 224x224.
    
    Returns:
        A PIL Image with random pixel values.
    """
    random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(random_array, mode='RGB')
    return img

def upload_image_to_s3(
    image: Image.Image,
    filename: str,
    bucket: str,
    prefix: str,
    s3_client
):
    """
    Save image to buffer and upload to S3.
    
    Args:
        image: PIL Image to upload
        filename: Target filename (e.g., 'img_0000_class_0.jpg')
        bucket: S3 bucket name
        prefix: S3 prefix/folder path
        s3_client: Boto3 S3 client
    """
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)
    s3_key = f"{prefix}/{filename}"
    s3_client.upload_fileobj(buffer, bucket, s3_key)

    
def main():
    bucket = "lab-genai-gal12012026"
    prefix = "benchmarks/s3loading/data"
    total_images = 1000
    num_classes = 10
    batch_size = 100
    max_workers = 10 
    
    s3_client = boto3.client('s3')
    
    current_batch = []  # Store (image, filename) tuples
    pbar = tqdm(total=total_images, desc="Uploading images")
    
    # Main loop with progress bar
    for i in range(total_images):
        class_id = i % num_classes
        filename = f"img_{i:04d}_class_{class_id}.jpg"
        
        # Generate image
        image = generate_random_image()
        
        # Add to current batch
        current_batch.append((image, filename))
        
        # When batch is full, upload concurrently
        if (i + 1) % batch_size == 0:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for image, fname in current_batch:
                    future = executor.submit(
                        upload_image_to_s3,
                        image, fname, bucket, prefix, s3_client
                    )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                        try:
                            future.result()
                            pbar.update(1)
                        except Exception as e:
                            print(f"Upload failed: {e}")
                            pbar.update(1)
            current_batch.clear()
            
if __name__ == "__main__":
    main()