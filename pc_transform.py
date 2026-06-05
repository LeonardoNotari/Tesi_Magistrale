import numpy as np
import pickle
import torch
import torch.nn.functional as func_F
import open_clip
import torch.optim as optim
from collections import OrderedDict
import ulip_models_data.models.ULIP_models as models
import argparse
from decoder_v1 import ULIPDecoder_v1  
from decoder_v2 import ULIPDecoder_v2
from decoder_v3 import TransformerDecoder
from tools.decoder_v4 import FoldingDecoder  
import os
import open3d as o3d
import trimesh
from pc_resample import PointCloudResampler

DATASET = 'datasets/objxl_animal/embeddings'
DATASET = 'datasets/animal_align/embeddings'

MODEL = 1 # 1 = MLP base, 2 = MLP v2, 3 = Transformer, 4 = Folding
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ULIP_CHECKPOINT    = "ulip_models_data/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
EMB_DIM = 1280    
SUBSAMPLING = 10000
N_POINTS = SUBSAMPLING
ADAPT_STEPS = 300
LR = 1e-5

PREFIX_TEXT = "a picture of"

#DIR = "transform_input_face/"
#DIR = "transform_input_spot/"
DIR = "transform_input_donkey/"

DECODER_CHECKPOINT = "checkpoints/decoder_animal_align.pth"  
#DECODER_CHECKPOINT = "checkpoints/decoder_allanimal.pth"  
#DECODER_CHECKPOINT = "checkpoints/decoder_objxl_animal.pth"  
#DECODER_CHECKPOINT = "checkpoints/decoder_objxl_v3_new__16_2048_35.pth" 
#DECODER_CHECKPOINT = "checkpoints/decoder_objxl_v3_new_32_2048.pth"  

#DECODER_CHECKPOINT = "checkpoints/decoder_coma_v3_new_32_2048.pth"  
#DECODER_CHECKPOINT = "checkpoints/decoder_coma.pth" 

ORIG_EMB = DIR + "original_embedding.npy" 
ORIG_PC  = DIR + "original_pointcloud.npy" 
GENERATED_PC = DIR + ((DECODER_CHECKPOINT.split('/')[1]).split('.')[0]) + '/' + "generated_pointcloud" + '_' + str(ADAPT_STEPS)    

TRASLATED_PC  =  ((DECODER_CHECKPOINT.split('/')[1]).split('.')[0]) + '/' 


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

def slerp(v0, v1, t):
    v0 = v0.squeeze()
    v1 = v1.squeeze()
    v0 = v0 / (np.linalg.norm(v0) + 1e-8)
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    omega = np.arccos(dot)          # angolo tra i due vettori
    
    if np.abs(omega) < 1e-6:        # quasi paralleli: fallback lineare
        return v0 + t * (v1 - v0)
    
    sin_omega = np.sin(omega)
    return (np.sin((1 - t) * omega) / sin_omega) * v0 + \
           (np.sin(t * omega)       / sin_omega) * v1

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




def infer_save(emb, model, output_pc, show):
    emb = torch.from_numpy(emb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(emb)   
    pred = pred.squeeze(0).cpu().numpy()  
    
    if show == True:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred)
        o3d.visualization.draw_geometries([pcd])
    
    save = input('Salvare la pointcloud ? [y/n]: ')
    if save == 'y':
        os.makedirs(DIR + ((DECODER_CHECKPOINT.split('/')[1]).split('.')[0]), exist_ok=True)
        np.save(output_pc + '.npy', pred)

    down = PointCloudResampler.voxel_downsample(pred, voxel_size=0.02)

    # Upsampling
    up = PointCloudResampler.upsample_merge(down, n_new=10000, method="jitter", voxel_size=0.03)
    if show == True:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(up)
        o3d.visualization.draw_geometries([pcd])
    
    save = input('Salvare la pointcloud ? [y/n]: ')
    if save == 'y':
        os.makedirs(DIR + ((DECODER_CHECKPOINT.split('/')[1]).split('.')[0]), exist_ok=True)
        np.save(output_pc + '_enhanced.npy', up)


def convert_to_npy(path_pc):
    obj = trimesh.load(path_pc)

    # Caso: PLY point cloud
    if isinstance(obj, trimesh.points.PointCloud):
        print("Rilevata point cloud PLY → converto in .npy")
        points = np.asarray(obj.vertices)

    # Caso: mesh (obj, ply, glb)
    elif isinstance(obj, trimesh.Trimesh):
        print("Rilevata mesh → eseguo sampling")
        points, _ = trimesh.sample.sample_surface(obj, N_POINTS)

    else:
        raise ValueError(f"Formato non supportato: {type(obj)}")
    points = points.astype(np.float32)
    # Salvataggio
    out_path = os.path.splitext(path_pc)[0] + ".npy"
    np.save(out_path, points)
    return points



def find_direction(p_pos, p_neg, orig_emb, mode):
    # 0: direzione calcolata con i due poli
    # 1: calcolo i baricentri dei punti più spostati verso i poli
    # 2: direzione tra polo positivo e embedding
    # 3: direzione tra baricentro dei punti più vicini al polo positivo e embedding
    # 4: direzione data da baricentro dei punti più vicini al polo positivo e quello dei punti più lontani dal polo positivo

    print(p_pos)
    print(orig_emb)
    if mode == '0':
        direction = p_pos - p_neg
    else:
        lis = []
        for  i in os.listdir(DATASET):
            j = np.load(DATASET + '/' + i)
            j = j / np.linalg.norm(j) 
            lis.append(j)
        pcs = np.array(lis)
        match mode :
            case '1':
                text_dir = p_pos - p_neg
                text_dir = text_dir / (np.linalg.norm(text_dir) + 1e-8)
                l = pcs @ text_dir
                order = np.argsort(l)[::-1]   # ordina decrescente
                embs = pcs[order]
                top_embs = embs[: int(len(lis)*0.05)]
                bot_embs = embs[- int(len(lis)*0.05):]
                e_pos = top_embs.mean(axis = 0)
                e_neg = bot_embs.mean(axis = 0)
                direction = e_pos - e_neg
            case '2':
                direction = p_pos - orig_emb
            case '3':
                l = pcs @ p_pos
                order = np.argsort(l)[::-1]   # ordina decrescente
                embs = pcs[order]
                top_embs = embs[: int(len(lis)*0.05)]
                e_pos = top_embs.mean(axis = 0)
                direction = e_pos - orig_emb
            case '4':
                l = pcs @ p_pos
                order = np.argsort(l)[::-1]   # ordina decrescente
                embs = pcs[order]
                top_embs = embs[: int(len(lis)*0.05)]
                bot_embs = embs[- int(len(lis)*0.05):]
                e_pos = top_embs.mean(axis = 0)
                e_neg = bot_embs.mean(axis = 0)
                direction = e_pos - e_neg

    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction

             


def generate_emb(model, path_pc):
    print("L'EMBEDDING DELLA PC DI ESEMPIO NON ESISTE → LO CALCOLO") 
    if path_pc.split('.')[-1] != 'npy':
        pc = convert_to_npy(path_pc)
    else:
        pc = np.load(path_pc).astype(np.float32) 
    if pc.shape[1] == 3:
        rgb = np.zeros_like(pc) 
        pc = np.concatenate([pc, rgb], axis=1) 
    pc_t = torch.from_numpy(pc).unsqueeze(0).to(DEVICE)  
    with torch.no_grad():
        embedding = model.encode_pc(pc_t)  
    embedding = func_F.normalize(embedding, dim=-1).squeeze(0)  
    embedding = embedding.cpu().numpy()
    return embedding



def text_emb(model, tokenizer, pole):
    prefix = PREFIX_TEXT
    text = f"{prefix} {pole}"
    with torch.no_grad():
        tokens = tokenizer(text).to(DEVICE)
        embs = model.encode_text(tokens)           # (2, D)
        emb = func_F.normalize(embs.float(), dim=-1)
    emb = emb.squeeze()
    t_pos = emb.cpu().numpy()
    return t_pos




def generate_direction(model, tokenizer, embedding, mode, input_poles):
    if (mode == '0' or mode == '1') and input_poles[1] == '':        
        print("le modalità 0 e 1 hanno bisogno di due poli")
        return None, model, tokenizer

    if os.path.exists('poles.pkl'):
        with open('poles.pkl', "rb") as f:
            poles = pickle.load(f)
    else:
        poles = {}
    p = []
    for i in [0,1]:
        if input_poles[i] == '' or input_poles[i] in poles or input_poles[i][:-4].replace('/','') in poles or f"{PREFIX_TEXT} {input_poles[i]}" in poles:
            emb = next((e for e in (
                poles.get(input_poles[i][:-4].replace('/', '')),
                poles.get(input_poles[i]),
                poles.get(f"{PREFIX_TEXT} {input_poles[i]}")
            ) if e is not None), None)
            p.append(emb)
        else:
            if model is None:
                model = load_model(ULIP_CHECKPOINT)
                tokenizer = open_clip.get_tokenizer("ViT-g-14")
                model.open_clip_model = model.open_clip_model.half()
            if os.path.exists(input_poles[i]):
                emb = generate_emb(model, input_poles[i])
                poles[input_poles[i][:-4].replace('/','')] = emb
            else:
                emb = text_emb(model, tokenizer, input_poles[i])
                poles[f"{PREFIX_TEXT} {input_poles[i]}"] = emb
            p.append(emb)
            with open('poles.pkl', "wb") as f:
                    pickle.dump(poles, f)
    direction = find_direction(p[0], p[1], embedding, mode)

    return direction, model, tokenizer
    


def main():
    if  not os.path.exists(ORIG_EMB) :
        model_ULIP = load_model(ULIP_CHECKPOINT)
        tokenizer = open_clip.get_tokenizer("ViT-g-14")
        model_ULIP.open_clip_model = model_ULIP.open_clip_model.half()
    else: 
        model_ULIP = None
        tokenizer = None
    
    if os.path.exists(ORIG_EMB): # se ho sia embedding che point cloud
        print("L'EMBEDDING DELLA PC ESISTE")
        embedding = np.load(ORIG_EMB).astype(np.float32)
    else:
        print("L'EMBEDDING DELLA PC NON ESISTE → LO CALCOLO") # se non ho l'embedding corrispondente al point cloud lo calcolo
        embedding = generate_emb(model_ULIP, ORIG_PC)
    

    match MODEL:
        case 1:
            model = ULIPDecoder_v1(emb_dim=1280, num_points=N_POINTS).to(DEVICE)
        case 2:
            model = ULIPDecoder_v2(emb_dim=1280, num_points=N_POINTS).to(DEVICE)
        case 3:
            if SUBSAMPLING == 5000:
                model = TransformerDecoder(D=1280, num_queries=100, points_per_query=50, num_blocks=2, dropout=0.3).to(DEVICE)  # 5000 punti 100 50 
            else:
                model = TransformerDecoder(D=1280, num_queries=64, points_per_query=32, num_blocks=2, dropout=0.3).to(DEVICE)  # 2048 punti
        case 4:
            model = FoldingDecoder(D=1280, N_seeds=32, K=64).to(DEVICE)      # 2048 punti

    model.load_state_dict(torch.load(DECODER_CHECKPOINT, map_location=DEVICE))

    model.eval()

    if ADAPT_STEPS > 0:

        print("Inizio adattamento del decoder al nuovo esempio...")
        model.train()
        opt = optim.Adam(model.parameters(), lr=LR)

        pc  = np.load(ORIG_PC ).astype(np.float32)

        emb = torch.from_numpy(embedding).unsqueeze(0).to(DEVICE)   
        pc  = torch.from_numpy(pc ).unsqueeze(0).to(DEVICE) 

        # addestramento sull'esempio da trasformare Test-Time Adaptation
        for i in range(ADAPT_STEPS):
            pred = model(emb)  

            pred_sub = subsample(pred, SUBSAMPLING)
            pc_sub   = subsample(pc,   SUBSAMPLING)

            loss = chamfer_distance(pred_sub, pc_sub)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i+1) % 50 == 0:
                print(f"  Iter {i+1}/{ADAPT_STEPS} - Loss: {loss.item():.6f}")

    model.eval()


    infer_save(embedding, model, GENERATED_PC, False)
    input_poles = []
    input_poles.append(input("inserire testo, nome o percorso della mesh per il polo positivo o invio per uscire "))
    input_poles.append(input("inserire testo, nome o percorso della mesh per il polo negativo o invio per avere un solo polo "))
    print(input_poles)
    while input_poles[0] != '':
        mode = input("inserire mode tra 0 e 4 o un altro tasto per uscire: ")
        while mode in ['0','1','2','3','4']:
            direction, model_ULIP, tokenizer = generate_direction(model_ULIP, tokenizer, embedding, mode, input_poles)
            if direction is not None:
                d_orth = direction - np.dot(direction, embedding) * embedding
                d_orth = d_orth / (np.linalg.norm(d_orth) + 1e-8)
                scale = 1.0

                i = input('Inserire valore di alpha tra -1 e 1, inserire qualsiasi altro valore per uscire: ')
                a = float(i)
                while a <= 1 and a >= -1:
                    if a >= 0:
                        q = np.cos(scale) * embedding + np.sin(scale) * d_orth
                        e_moved = slerp(embedding, q, a)        # verso d
                    else:
                        q = np.cos(scale)*embedding - np.sin(scale) * d_orth
                        e_moved = slerp(embedding, q, -a)            
                    infer_save(e_moved.astype(np.float32), model, (DIR+TRASLATED_PC+input_poles[0]+'_'+input_poles[1]+'_'+str(ADAPT_STEPS)+'_'+i+'_mode_'+mode), False)
                    i = input('Inserire valore di alpha tra -1 e 1, inserire qualsiasi altro valore per uscire: ')
                    a = float(i)
            mode = input("inserire mode tra 0 e 4 o un altro tasto per uscire: ")
        input_poles = []
        input_poles.append(input("inserire testo, nome o percorso della mesh per il polo positivo o invio per uscire "))
        input_poles.append(input("inserire testo, nome o percorso della mesh per il polo negativo o invio per avere un solo polo "))


if __name__ == "__main__":
    main()



