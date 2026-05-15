from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from dataloader import ULIPPointCloudDataset
from decoder import ULIPDecoder
#from decoder_v2 import ULIPDecoder
import os
import matplotlib.pyplot as plt




def chamfer_distance(x, y):
    # x: (B, N, 3)
    # y: (B, M, 3)
    x_exp = x.unsqueeze(2)  # (B, N, 1, 3)
    y_exp = y.unsqueeze(1)  # (B, 1, M, 3)

    dist = torch.sum((x_exp - y_exp)**2, dim=3)  # (B, N, M)

    min_x, _ = torch.min(dist, dim=2)  # (B, N)
    min_y, _ = torch.min(dist, dim=1)  # (B, M)

    return min_x.mean() + min_y.mean()

def subsample(pc, k):
    idx = torch.randperm(pc.shape[1], device=pc.device)[:k]
    return pc[:, idx]




# cartelle dataset
PC_FOLDER = "datasets/objxl_animal/point_clouds"
EMB_FOLDER = "datasets/objxl_animal/embeddings"
CKPT_PATH = "checkpoints/decoder_objxl_animal_v1.pth"

dataset = ULIPPointCloudDataset(PC_FOLDER, EMB_FOLDER)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ULIPDecoder(emb_dim=1280, num_points=10000).to(device)
opt = optim.Adam(model.parameters(), lr=5e-4)

EPOCHS = 100

os.makedirs("checkpoints", exist_ok=True)
best_loss = float("inf")
loss_ = []
epochs_ = []
print('inizio training')
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for emb, pc in loader:
        emb = emb.to(device)
        pc = pc.to(device)

        pred = model(emb)

        pred_sub = subsample(pred, 2048)
        pc_sub   = subsample(pc, 2048)

        loss = chamfer_distance(pred_sub, pc_sub)

        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    loss_.append(epoch_loss)
    epochs_.append(epoch)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")
    if epoch_loss < best_loss - 1e-4:
        best_loss = epoch_loss
        torch.save(model.state_dict(), CKPT_PATH)
        print("→ Salvato nuovo modello migliore!")

plt.plot(epochs_,loss_)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss')
