from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from dataloader import ULIPPointCloudDataset
from decoder_v1 import ULIPDecoder_v1
from decoder_v2 import ULIPDecoder_v2
from decoder_v3 import TransformerDecoder
from tools.decoder_v4 import FoldingDecoder
import os
import matplotlib.pyplot as plt
from torch.utils.data import random_split


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
MODEL = 1 # 1 = MLP base, 2 = MLP v2, 3 = Transformer
PC_FOLDER = "datasets/shapenet_plane/point_clouds"
EMB_FOLDER = "datasets/shapenet_plane/embeddings"
CKPT_PATH = "checkpoints/decoder_shapenet_plane_v" + str(MODEL) 
BATCH = 32
SUBSAMPLING = 2048
NUM_POINTS = 2048
TRAIN_RATIO = 0.85
EPOCHS = 30




dataset = ULIPPointCloudDataset(PC_FOLDER, EMB_FOLDER)

total = len(dataset)
train_size = int(TRAIN_RATIO * total)
val_size = total - train_size 
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=8)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

match MODEL:
    case 1:
        model = ULIPDecoder_v1(emb_dim=1280, num_points=NUM_POINTS).to(device)
    case 2:
        model = ULIPDecoder_v2(emb_dim=1280, num_points=NUM_POINTS).to(device)
    case 3:
        if SUBSAMPLING == 2048:
            model = TransformerDecoder(D=1280, num_queries=64, points_per_query=32, num_blocks=2, dropout=0.3).to(device)  # 2048 punti
        elif SUBSAMPLING == 5000:
            model = TransformerDecoder(D=1280, num_queries=64, points_per_query=64, num_blocks=2, dropout=0.3).to(device)  # 4096 punti
    case 4:
        model = FoldingDecoder(D=1280, N_seeds=32, K=64).to(device)      # 2048 punti


opt = optim.Adam(model.parameters(), lr=1e-5, weight_decay = 1e-3)



os.makedirs("checkpoints", exist_ok=True)
best_loss = float("inf")
val_loss_ = []
loss_ = []
epochs_ = []


print('inizio training')
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    
    for emb, pc in train_loader:
        emb = emb.to(device)
        pc = pc.to(device)

        pred = model(emb)

        pred_sub = subsample(pred, SUBSAMPLING)
        pc_sub   = subsample(pc, SUBSAMPLING)

        loss = chamfer_distance(pred_sub, pc_sub)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # aggiungi questa
        opt.step()
        train_loss += loss.item()
    

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for emb, pc in val_loader:
            emb, pc = emb.to(device), pc.to(device)
            pred = model(emb)
            pred_sub = subsample(pred, SUBSAMPLING)
            pc_sub   = subsample(pc,   SUBSAMPLING)
            val_loss += chamfer_distance(pred_sub, pc_sub).item()
    
    val_loss /= len(val_loader)
    val_loss_.append(val_loss)
    train_loss /= len(train_loader)
    loss_.append(train_loss)
    epochs_.append(epoch)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    
    if val_loss < best_loss - 1e-5:
        best_loss = val_loss
        torch.save(model.state_dict(), CKPT_PATH + '_' + str(BATCH) + '_' + str(SUBSAMPLING) + '.pth')
        best_epoch = epoch
        print("→ Salvato nuovo modello migliore!")
    plt.plot(epochs_,loss_,'r')
    plt.plot(epochs_,val_loss_,'b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(CKPT_PATH)
    plt.savefig('losses/loss_' + (CKPT_PATH.split('/')[1]) + '_' + str(BATCH) + '_' + str(SUBSAMPLING))
print(best_epoch)
    

plt.plot(epochs_,loss_,'r', label = 'train_loss')
plt.plot(epochs_,val_loss_,'b', label = 'val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(CKPT_PATH)
plt.legend()
plt.savefig('losses/loss_' + (CKPT_PATH.split('/')[1]) + '_' + str(BATCH) + '_' + str(SUBSAMPLING))
