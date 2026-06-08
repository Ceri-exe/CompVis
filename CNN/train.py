import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np
from model import CNN

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def plot_first_layer_filters(model): 
    """Plots the weights/filters learned by the first convolutional layer."""
    W = model.state_dict()['conv1.0.weight'].data.cpu()
    w_min, w_max = W.min().item(), W.max().item()
    fig, axes = plt.subplots(3, 6)
    fig.subplots_adjust(hspace=0.01, wspace=0.1)
    
    for i, ax in enumerate(axes.flat):
        if i < 16:
            ax.set_xlabel(f"Filter: {i}")
            ax.imshow(W[i, :].view(5, 5), vmin=w_min, vmax=w_max, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if i in range(16, 18):
            ax.axis('off')
    plt.show()

def train_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0
    
    for batch_nr, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, yhat = torch.max(pred.data, 1)
        correct += (yhat == y).sum().item()
        train_loss += loss.item()

        if batch_nr % 100 == 0:
            current = (batch_nr + 1) * len(X)
            print(f"Loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
            
    return train_loss / size, correct / size

def validate(dataloader, model, loss_fn, device):
    model.eval()
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    validation_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            validation_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    avg_loss = validation_loss / num_batches
    accuracy = correct / size
    print(f"Validation Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return avg_loss, accuracy

def show_mismatched_data(dataset, model, device, num_images=5):
    """Evaluates individual samples to visualize target failure modes."""
    model.eval()
    softmax_fn = nn.Softmax(dim=1)
    count = 0
    
    print("\n--- Evaluating Sample Misclassifications ---")
    with torch.no_grad():
        for X, y in dataset:
            X_input = X.reshape(1, 1, 28, 28).to(device)
            z = model(X_input)
            _, yhat = torch.max(z, 1)
            yhat = int(yhat.item())
            
            if yhat != y:
                print(f"Mismatch -> True Class: {y} | Predicted: {yhat} | Confidence: {torch.max(softmax_fn(z)).item():.4f}")
                count += 1
            if count >= num_images:
                break

if __name__ == "__main__":
    n_epochs = 5
    learning_rate = 0.001
    batch_size_train = 100
    batch_size_val = 500

    device = get_device()
    print(f"Running on: {device}")

    # Dataset pipelines
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
    validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size_val, shuffle=True, num_workers=1)

    # Initialize model network configurations
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Execute runtime configurations
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        t_loss, t_acc = train_epoch(train_loader, model, loss_func, optimizer, device)
        v_loss, v_acc = validate(validation_loader, model, loss_func, device)
        
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        
    print("Done Training!")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Display learned features and failure modes
    plot_first_layer_filters(model)
    show_mismatched_data(validation_dataset, model, device)