import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# 1. Custom Dataset Loader
class F1TenthDataset(Dataset):
    def __init__(self, npz_path="f1tenth_dataset.npz"):
        data = np.load(npz_path)
        # Normalize scans from [0, 10] meters to [0, 1] for stable gradient descent
        scans = data["scans"] / 10.0
        actions = data["actions"]

        self.scans = torch.tensor(scans, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        return self.scans[idx], self.actions[idx]

# 2. MLP Policy Network
class DrivingPolicyNet(nn.Module):
    def __init__(self, input_dim=1080, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and create 80/20 train/validation split
    full_dataset = F1TenthDataset("f1tenth_dataset.npz")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Initialize model, loss function (MSE for regression), and Adam optimizer
    model = DrivingPolicyNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 25
    print("Beginning training...")

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        for scans_batch, actions_batch in train_loader:
            scans_batch = scans_batch.to(device)
            actions_batch = actions_batch.to(device)

            # Forward pass
            preds = model(scans_batch)
            loss = criterion(preds, actions_batch)

            # Backward pass & optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(scans_batch)

        train_loss /= train_size

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for scans_batch, actions_batch in val_loader:
                scans_batch = scans_batch.to(device)
                actions_batch = actions_batch.to(device)
                preds = model(scans_batch)
                loss = criterion(preds, actions_batch)
                val_loss += loss.item() * len(scans_batch)

        val_loss /= val_size

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

    # Save trained weights
    torch.save(model.state_dict(), "f1tenth_policy.pth")
    print("Training complete! Model saved to 'f1tenth_policy.pth'")

if __name__ == "__main__":
    train()