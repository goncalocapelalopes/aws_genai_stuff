import argparse
import torch
import pynvml

from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from s3dataset import S3ImageDataset
from model import TrivialCNN
from train_utils import train_epoch
def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Data configuration
    parser.add_argument('--bucket', type=str, default='lab-genai-gal12012026')
    parser.add_argument('--prefix', type=str, default='benchmarks/s3loading/data')
    parser.add_argument('--num-images', type=int, default=1000)
    parser.add_argument('--num-classes', type=int, default=10)
    
    return parser.parse_args()

def main():
    
    args = parse_args()
    print(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, "
      f"num_workers={args.num_workers}, lr={args.learning_rate}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type != 'cuda':
        print("WARNING: CUDA not available, aborting training.")
        return
    
    # Initialize GPU monitoring
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataset and dataloader
    dataset = S3ImageDataset(
        bucket=args.bucket,
        prefix=args.prefix,
        num_images=args.num_images,
        num_classes=args.num_classes,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    print(f"Dataset size: {len(dataset)} images")
    print(f"Number of batches: {len(dataloader)}")
    
    model = TrivialCNN(num_classes=args.num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        throughput, avg_loss = train_epoch(
            model, dataloader, criterion, optimizer, 
            device, epoch + 1, gpu_handle
        )
    
    pynvml.nvmlShutdown()
    print("Training complete!")

if __name__ == "__main__":
    main()