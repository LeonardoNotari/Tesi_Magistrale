import os
import torch
from torch.utils.data import Dataset
import numpy as np

class ULIPPointCloudDataset(Dataset):
    def __init__(self, pc_folder, emb_folder):
        self.pc_folder = pc_folder
        self.emb_folder = emb_folder

        # lista dei nomi senza estensione 
        self.names = [f[:-4] for f in os.listdir(pc_folder) if f.endswith(".npy")]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        pc = np.load(os.path.join(self.pc_folder, name + ".npy")).astype(np.float32)
        emb = np.load(os.path.join(self.emb_folder, name + ".npy")).astype(np.float32)

        pc = torch.from_numpy(pc)          # (N, 3)
        emb = torch.from_numpy(emb)        # (D,)

        return emb, pc
