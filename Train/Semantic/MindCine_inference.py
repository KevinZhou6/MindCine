import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import os
from loss import ClipLoss,soft_clip_loss,GroupLoss
class MindCine(nn.Module):
    def __init__(self, h=10, out_dim=77*768, seq_len=62, n_blocks=2, drop=0., clip_size=4096, blurry_recon=True, clip_scale=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(310, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10000),
            nn.ReLU(),
            nn.Linear(10000, 768),
           
            
        )

        self.align = nn.Sequential(
            nn.Linear(768, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),

            nn.Linear(10000, out_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cliploss = ClipLoss()
            
    
        
    def forward(self, x):
        x1 = self.mlp(x)
        x2 = self.align(x1)
        return x1,x2


class Dataset():
    def __init__(self, eeg, text,img_emb,text_emb):

        # scaler = preprocessing.StandardScaler().fit(eeg)
        # eeg = scaler.transform(eeg)

        self.eeg = eeg
        self.text = text
        self.img_emb = img_emb
        self.text_emb = text_emb
        self.len = eeg.shape[0]


    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.text[item], self.img_emb[item], self.text_emb[item]

GT_label = np.array([[23, 22, 9, 6, 18,       14, 5, 36, 25, 19,      28, 35, 3, 16, 24,      40, 15, 27, 38, 33,
             34, 4, 39, 17, 1,       26, 20, 29, 13, 32,     37, 2, 11, 12, 30,      31, 8, 21, 7, 10, ],
            [27, 33, 22, 28, 31,     12, 38, 4, 18, 17,      35, 39, 40, 5, 24,      32, 15, 13, 2, 16,
 	         34, 25, 19, 30, 23,     3, 8, 29, 7, 20,        11, 14, 37, 6, 21,      1, 10, 36, 26, 9, ],
            [15, 36, 31, 1, 34,      3, 37, 12, 4, 5,        21, 24, 14, 16, 39,     20, 28, 29, 18, 32,
             2, 27, 8, 19, 13,       10, 30, 40, 17, 26,     11, 9, 33, 25, 35,      7, 38, 22, 23, 6,],
            [16, 28, 23, 1, 39,      10, 35, 14, 19, 27,     37, 31, 5, 18, 11,      25, 29, 13, 20, 24,
            7, 34, 26, 4, 40 ,       12, 8, 22, 21, 30,      17, 2, 38, 9,  3 ,      36, 33, 6, 32, 15,],
            [18, 29, 7, 35, 22  ,    19, 12, 36, 8, 15,      28, 1, 34, 23, 20 ,     13, 37, 9, 16, 30  ,
             2, 33, 27, 21, 14 ,     38, 10, 17, 31, 3,      24, 39, 11, 32, 4,      25, 40, 5, 26, 6 ,],
            [29, 16, 1, 22, 34,      39, 24, 10, 8, 35,      27, 31, 23, 17, 2,      15, 25, 40, 3, 36,
             26, 6, 14, 37, 9,       12, 19, 30, 5, 28,      32, 4, 13, 18, 21,      20, 7, 11, 33, 38],
            [38, 34, 40, 10, 28,     7, 1, 37, 22, 9,        16, 5, 12, 36, 20,      30, 6, 15, 35, 2,
             31, 26, 18, 24, 8,      3, 23, 19, 14, 13,      21, 4, 25, 11, 32,      17, 39, 29, 33, 27]
            ])


chosed_label = [i for i in range(1,41)]

device='cuda:3'
import datetime
def get_time():
    current_time = datetime.datetime.now()
    new_hour = (current_time.hour + 8) % 24
    new_time = current_time.replace(hour=new_hour)
    formatted_time = new_time.strftime("%m-%d_%H-%M")
    return formatted_time

if __name__ == '__main__':
    eegdata = np.load('data/SEED-DV/DE_1per2s/sub1.npy')
    print(eegdata.shape)
    eeg=[]
    for i in range(6):
        indices = [list(GT_label[i]).index(element) for element in chosed_label]
        chosed_eeg = eegdata[i][indices,:]
        eeg.append(chosed_eeg)
    eeg = np.stack(eeg,axis=0)
    eeg = torch.from_numpy(eeg)
    eeg = rearrange(eeg, 'a b c e f -> (a b c) (e f)')
    test_indices = [list(GT_label[6]).index(element) for element in chosed_label]
    test_eeg = eegdata[6][indices,:]
    test_eeg = rearrange(test_eeg, 'b c e f -> (b c) (e f)')
    
    # EEG = torch.mean(EEG, dim=1).resize(EEG.shape[0], 310)
    # print(EEG.shape)
    # print(Text.shape)
    model_file='40classes-sub1.pt'
    model = MindCine()
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)['state_dict'])
    
    model=model.to(device)
    normalize = preprocessing.StandardScaler()
    normalize.fit(eeg)
    eeg = normalize.transform(eeg)
    test_eeg = normalize.transform(test_eeg)
    test_eeg = torch.from_numpy(test_eeg)
    print(test_eeg.shape)
    pool_embeddings,test_embeddings =model(test_eeg.float().to(device))
    test_embeddings = test_embeddings.reshape(-1,77,768)
    print(test_embeddings.shape)
    
    path = f'text_embeds/MindCine/' 
    os.makedirs(path,exist_ok=True)

    
    torch.save(test_embeddings, path + 'block7_40classes-sub1.pt')
    

    



