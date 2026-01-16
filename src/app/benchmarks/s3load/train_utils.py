import time
import psutil
import torch
import pynvml

def train_epoch(model, dataloader, criterion, optimizer, device, epoch_num, gpu_handle):
    model.train()
    total_images = 0
    epoch_start = time.time()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        batch_start = time.time()
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        batch_time = time.time() - batch_start
        batch_throughput = len(images) / batch_time
        total_images += len(images)
        total_loss += loss.item()
        
        # Log every 10 batches
        if batch_idx % 10 == 0:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent()
            ram_gb = psutil.virtual_memory().used / 1024**3
            gpu_memory_gb = torch.cuda.memory_allocated(device) / 1024**3
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
            
            print(f"Batch {batch_idx}: {batch_throughput:.2f} img/s | "
                  f"Loss: {loss.item():.4f} | "
                  f"CPU: {cpu_percent:.1f}% | "
                  f"RAM: {ram_gb:.2f}GB | "
                  f"GPU Mem: {gpu_memory_gb:.2f}GB | "
                  f"GPU Util: {gpu_util}%")

    epoch_time = time.time() - epoch_start
    epoch_throughput = total_images / epoch_time
    avg_loss = total_loss / len(dataloader)
    
    print(f"\nEpoch {epoch_num} Summary: {epoch_throughput:.2f} img/s | Avg Loss: {avg_loss:.4f}\n")
    
    return epoch_throughput, avg_loss