import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np
from model import MLP

# --- Helper Functions ---
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def show_mismatched_data(dataloader, model, device, num_images=5):
    """Visualizes samples the model struggled to predict correctly."""
    model.eval()
    softmax = nn.Softmax(dim=1)
    count = 0
    
    print("\n--- Visualizing Sample Misclassifications ---")
    with torch.no_grad():
        for X, y in dataloader.dataset:
            X_dev = X.to(device).unsqueeze(0)
            z = model(X_dev)
            _, yhat = torch.max(z, 1)
            yhat = int(yhat.item())
            
            if yhat != y:
                plt.figure()
                plt.imshow(X.numpy().squeeze(), cmap='gray')
                plt.title(True Class: {y} | Predicted: {yhat})
                plt.axis('off')
                plt.show()
                
                print(f"Mismatch: True={y}, Predicted={yhat}, Confidence={torch.max(softmax(z)).item():.4f}")
                count += 1
            if count >= num_images:
                break

def train_epoch(dataloader, model, loss_fn, optimizer, device, input_dim):
    model.train()
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0
    
    for batch_nr, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.view(-1, input_dim))
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Tracking metrics
        _, yhat = torch.max(pred.data, 1)
        correct += (yhat == y).sum().item()
        train_loss += loss.item()

        if batch_nr % 100 == 0:
            current = (batch_nr + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
            
    return train_loss / len(dataloader), correct / size

def validate(dataloader, model, loss_fn, device, input_dim):
    model.eval()
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    validation_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.view(-1, input_dim))
            validation_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    avg_loss = validation_loss / num_batches
    accuracy = correct / size
    print(f"Validation Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return avg_loss, accuracy

# --- Main Execution Loop ---
if __name__ == "__main__":
    # Hyperparameters & Dimensions
    input_dim = 28 * 28
    hidden_dim = 15
    output_dim = 10
    n_epochs = 10
    learning_rate = 0.1
    batch_size = 100

    device = get_device()
    print(f"Using device: {device}")

    # Data Preparation
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
    validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Initialization
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    # Metrics history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Training Loop
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        t_loss, t_acc = train_epoch(train_loader, model, loss_func, optimizer, device, input_dim)
        v_loss, v_acc = validate(validation_loader, model, loss_func, device, input_dim)
        
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        
    print("Training Complete!")

    # Plotting training dynamics
    fig, ax1 = plt.subplots()
    ax1.plot(history["train_loss"], color='tab:red', label='Train Loss')
    ax1.plot(history["val_loss"], color='purple', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    
    ax2 = ax1.twinx()  
    ax2.plot(history["train_acc"], color='tab:blue', label='Train Acc')
    ax2.plot(history["val_acc"], color='tab:orange', label='Val Acc')
    ax2.set_ylabel('Accuracy', color='tab:blue')
    
    fig.tight_layout()
    plt.title("Training Metrics Over Time")
    plt.show()

    # Visualize failure modes
    show_mismatched_data(validation_loader, model, device)