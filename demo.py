import torch
import torch.nn as nn
import torch.optim as optim

# 1. Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Dummy data on chosen device
x = torch.randn(1024, 100, device=device)  # 1024 samples, 100 features
y = torch.randn(1024, 1, device=device)    # Targets

# 3. Simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

model = SimpleNet().to(device)  # move model to GPU
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training loop
for epoch in range(10):
    optimizer.zero_grad()
    preds = model(x)           # runs on GPU
    loss = criterion(preds, y)
    loss.backward()            # GPU gradients
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

