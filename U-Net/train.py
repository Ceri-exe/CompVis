import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from model import UNet
from dataset import get_dataloaders

# Global runtime configuration
# This might differ from dataset to dataset depending on how the extracted folders are named.
config = {
    "lr": 1e-3,
    "batch_size": 16,
    "image_dir": "data/images",
    "segmentation_dir": "data/segmentations",
    "image_paths": "data/images.txt",
    "epochs": 5,
    "checkpoint": "checkpoint/bird_segmentation_v1.pth",
    "optimiser": "checkpoint/bird_segmentation_v1_optim.pth",
    "continue_train": False,
    "device": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
}

def check_accuracy_and_save(model, optimiser, val_loader, epoch, device):
    """Evaluates the model on the validation set using the Dice Score and saves weights."""
    os.makedirs("test/pred", exist_ok=True)
    os.makedirs("test/true", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)
    
    torch.save(model.state_dict(), config['checkpoint'])
    torch.save(optimiser.state_dict(), config['optimiser'])

    dice_score = 0
    model.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            
            # Use autocast during validation inference as well
            device_type = "cuda" if "cuda" in device else "cpu"
            with torch.amp.autocast(device_type=device_type):
                preds = torch.sigmoid(model(x))
                
            preds = (preds > 0.5).float()
            
            # Dice Coefficient Calculation
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            
            # Save a few sample images from the first batch to visually track progress
            if i == 0:
                torchvision.utils.save_image(preds, f"test/pred/epoch_{epoch}.png")
                torchvision.utils.save_image(y, f"test/true/epoch_{epoch}.png")

    print(f"--- Epoch {epoch} Evaluation | Validation Average Dice Score: {dice_score/len(val_loader):.4f} ---\n")

def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    """Runs a single training epoch utilizing Automatic Mixed Precision (AMP)."""
    model.train()
    running_loss = 0.0
    
    # Initialize Gradient Scaler for AMP (Handles dynamic FP16 scaling)
    device_type = "cuda" if "cuda" in device else "cpu"
    scaler = torch.amp.GradScaler(device_type)
    
    loop = tqdm(dataloader, desc="Training Batches")
    for batch_idx, (image, seg) in enumerate(loop):
        image = image.to(device)
        seg = seg.float().to(device)

        # Forward pass wrapped in autocast for accelerated execution path
        with torch.amp.autocast(device_type=device_type):
            pred = model(image)
            loss = loss_fn(pred, seg)

        optimizer.zero_grad()
        
        # Backward pass using the scaled loss matrix
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        
        # Dynamically updates the TQDM bar with rolling averages
        loop.set_postfix(batch_loss=loss.item(), rolling_loss=(running_loss / (batch_idx + 1)))
        
    return running_loss / len(dataloader)

if __name__ == "__main__":
    print(f"Running execution pipeline on: {config['device']}")

    # Setup matching transformation structures
    transforms_image = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1., 1., 1.))
    ])
    transforms_mask = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,))
    ])

    # Fetch data streams
    train_loader, val_loader = get_dataloaders(config, [transforms_image, transforms_mask])
    print(f"Loaded: {len(train_loader)} training batches | {len(val_loader)} validation batches.")
    
    # Model Configurations
    model = UNet().to(config['device'])
    optimiser = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_func = nn.BCEWithLogitsLoss()

    # Optional checkpoint recovery
    if config['continue_train'] and os.path.exists(config['checkpoint']):
        print("Recovering weights from previous training checkpoint...")
        model.load_state_dict(torch.load(config['checkpoint'], map_location=config['device']))
        optimiser.load_state_dict(torch.load(config['optimiser'], map_location=config['device']))

    # Total network parameters audit check
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable U-Net Parameters: {total_params:,}")

    # Main Training Execution Loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 30)
        avg_train_loss = train_one_epoch(train_loader, model, loss_func, optimiser, config['device'])
        check_accuracy_and_save(model, optimiser, val_loader, epoch+1, config['device'])
        
    print("Training Pipeline Successfully Completed!")