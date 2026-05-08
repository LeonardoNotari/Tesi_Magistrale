import numpy as np
import pickle
import torch
import torch.nn.functional as func_F
import open_clip
import torch.optim as optim
from collections import OrderedDict
import ulip_models_data.models.ULIP_models as models
import argparse
from decoder import ULIPDecoder   
import os
import open3d as o3d




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ULIP_CHECKPOINT    = "ulip_models_data/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
EMB_DIM = 1280    
N_POINTS = 10000   

ALPHA = [-0.7, 0.7]
ADAPT_STEPS = 300
LR = 1e-5

PREFIX_TEXT = "a picture of "
PAIR = ("short",    "tall")

DECODER_CHECKPOINT = "checkpoints/decoder_animal_5000_align.pth"  
#DECODER_CHECKPOINT = "checkpoints/decoder_coma.pth"  
DIR = "trasform_input_donkey/"

ORIG_EMB = DIR + "original_embedding.npy" 
ORIG_PC  = DIR + "original_pointcloud.npy" 
GENERATED_PC = DIR + "generated_pointcloud" + '_' + str(ADAPT_STEPS) + ".npy"   
#TRASLATED_EMB = DIR + "traslated_embedding.npy" 
TRASLATED_PC  = PAIR[0] + '_' + PAIR[1] + '_' + str(ADAPT_STEPS) + ".npy"  



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

# CARICAMENTO MODELLO

def load_model(checkpoint_path):
    print(f"Caricamento modello da {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = OrderedDict() 
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v 
    args = argparse.Namespace(npoints=N_POINTS, use_height=False, gpu=0)
    model = models.ULIP2_PointBERT_Colored(args=args)
    model.to(DEVICE)
    model.load_state_dict(state_dict, strict=False) #carica i pesi nel modello
    model.eval()
    print("Modello caricato!")
    return model




# ENCODE COPPIA TESTO E CALCOLO DIREZIONE

def text_to_direction(model, tokenizer, text_pos, text_neg, prefix = PREFIX_TEXT):
    """
    Dati due label testuali, restituisce la direzione semantica normalizzata
    nello spazio ULIP: d = normalize(t_pos - t_neg)
    """
    texts = [f"{prefix} {text_pos}", f"{prefix} {text_neg}"]
    with torch.no_grad():
        tokens = tokenizer(texts).to(DEVICE)
        embs = model.encode_text(tokens)           # (2, D)
        embs = func_F.normalize(embs.float(), dim=-1)
    t_pos = embs[0].cpu().numpy()
    t_neg = embs[1].cpu().numpy()
    direction = t_pos - t_neg
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction, t_pos, t_neg





# ANALISI DATO UN TESTO POS/NEG

def move_emb(model, tokenizer, text_pos, text_neg):

    print(f"\n{'='*60}")
    print(f"Analisi direzione:  [{text_pos}]  →  [{text_neg}]")

    if os.path.exists('directions_txt.pkl'):
        with open('directions_txt.pkl', "rb") as f:
            directions = pickle.load(f)
    else:
        directions = []

    pairs=[]
    for i in directions:
        pairs.append(i[0])

    if (PREFIX_TEXT + PAIR[0],PREFIX_TEXT + PAIR[1]) not in pairs : # se ho già codificato la direzione la prendo da directions_text.pkl
        # calcola punti nell'ipersfera dei due testi e la direzione
        direction, t_pos, t_neg = text_to_direction(model, tokenizer, text_pos, text_neg)
        sim_tt = float(np.dot(t_pos, t_neg))
        directions.append([ (PREFIX_TEXT + PAIR[0],PREFIX_TEXT + PAIR[1]) , direction, sim_tt])
        with open('directions_txt.pkl', "wb") as f:
            pickle.dump(directions, f)
    else:
        for i in directions:
            if i[0] == (PREFIX_TEXT + PAIR[0],PREFIX_TEXT + PAIR[1]):
                direction = i[1]
                sim_tt = i[2]

    print(f"  cos_sim(t+, t-)  = {sim_tt:.4f}  ")
    
    if os.path.exists(ORIG_EMB): # se ho sia embedding che point cloud
        print("L'EMBEDDING DELLA PC ESISTE")
        embedding = np.load(ORIG_EMB).astype(np.float32)
    else:
        print("L'EMBEDDING DELLA PC NON ESISTE → LO CALCOLO") # se non ho l'embedding corrispondente al point cloud lo calcolo
        pc = np.load(ORIG_PC).astype(np.float32) 
        if pc.shape[1] == 3:
            rgb = np.zeros_like(pc) 
            pc = np.concatenate([pc, rgb], axis=1) 
        pc_t = torch.from_numpy(pc).unsqueeze(0).to(DEVICE)  
        with torch.no_grad():
            embedding = model.encode_pc(pc_t)  
        embedding = func_F.normalize(embedding, dim=-1).squeeze(0)  
        embedding = embedding.cpu().numpy()
        np.save(ORIG_EMB, embedding)
    embs_moved = []
    for a in ALPHA: 
        e_moved = embedding + a * direction # mi sposto del valore di alpha nella direzione
        e_moved = e_moved / (np.linalg.norm(e_moved) + 1e-8)
        embs_moved.append(e_moved)
    return embs_moved




def main():
    if os.path.exists('directions_txt.pkl'):
        with open('directions_txt.pkl', "rb") as f:
            directions = pickle.load(f)
    else:
        directions = [[]]
    pairs=[]
    for i in directions:
        pairs.append(i[0])

    #controllo se ho già l'embedding della point cloud da trasformare e la codifica della direzione per evitare di caricare il checkpoint di ULIP2

    if os.path.exists(ORIG_EMB) and (PREFIX_TEXT + PAIR[0],PREFIX_TEXT + PAIR[1]) in pairs: 
        model_ULIP = None
        tokenizer = None
    else:
        model_ULIP = load_model(ULIP_CHECKPOINT)
        tokenizer = open_clip.get_tokenizer("ViT-g-14")
        model_ULIP.open_clip_model = model_ULIP.open_clip_model.half()

    text_pos, text_neg = PAIR
    traslated_emb = move_emb(model_ULIP, tokenizer, text_pos, text_neg)
 
    #np.save(TRASLATED_EMB, traslated_emb)

    model = ULIPDecoder(emb_dim=EMB_DIM, num_points=N_POINTS).to(DEVICE)
    model.load_state_dict(torch.load(DECODER_CHECKPOINT, map_location=DEVICE))
    
    if ADAPT_STEPS > 0:
        print("Inizio adattamento del decoder al nuovo esempio...")
        model.train()
        opt = optim.Adam(model.parameters(), lr=LR)

        emb = np.load(ORIG_EMB).astype(np.float32)
        pc  = np.load(ORIG_PC ).astype(np.float32)

        emb = torch.from_numpy(emb).unsqueeze(0).to(DEVICE)   
        pc  = torch.from_numpy(pc ).unsqueeze(0).to(DEVICE) 

        # addestramento sull'esempio da trasformare Test-Time Adaptation
        for i in range(ADAPT_STEPS):
            pred = model(emb)  

            pred_sub = subsample(pred, 5000)
            pc_sub   = subsample(pc,   5000)

            loss = chamfer_distance(pred_sub, pc_sub)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i+1) % 50 == 0:
                print(f"  Iter {i+1}/{ADAPT_STEPS} - Loss: {loss.item():.6f}")

    model.eval()

    orig_emb = np.load(ORIG_EMB).astype(np.float32)
    orig_emb = torch.from_numpy(orig_emb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(orig_emb)   
    pred = pred.squeeze(0).cpu().numpy()   
    np.save(GENERATED_PC, pred)

    #tras_emb = np.load(TRASLATED_EMB).astype(np.float32)
    i = 0
    for tras_emb in traslated_emb:
        tras_emb = torch.from_numpy(tras_emb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(tras_emb)   
        pred = pred.squeeze(0).cpu().numpy()   
        np.save(DIR + str(ALPHA[i]) + '_' + TRASLATED_PC, pred) 
        i += 1


if __name__ == "__main__":
    main()



