import random 
import os
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import os 
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
from einops import rearrange
from sklearn.preprocessing import StandardScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint

from Modules.BIOT.biot import (
    BIOTEncoder,
)
import torch
from loss import ClipLoss,soft_clip_loss,GroupLoss



class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, pretrain_model_choice = 0):
        super().__init__() 
        pretrain_models = ["Modules/BIOT/EEG-six-datasets-18-channels.ckpt"]
        in_channels=18
        # if pretrain_model_choice == 0: in_channels = 16
        # elif pretrain_model_choice == 1: in_channels = 18
        # elif pretrain_model_choice == 2: in_channels = 18
        # else: raise ValueError("pretrain_model_choice should be 0, 1, or 2")
        
        self.chan_conv      = Conv1dWithConstraint(62, in_channels, 1, max_norm=1)
        model = BIOTEncoder(emb_size=256, heads=8, depth=4)
        model.load_state_dict(torch.load(pretrain_models[pretrain_model_choice]))
        print(f"load pretrain model from {pretrain_models[pretrain_model_choice]}")
        for p in model.parameters():
            p.requires_grad = False
        self.feature        = model
        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
        )
        self.align = nn.Sequential(
            nn.Linear(768, 10000),
            nn.ReLU(),
            # nn.Linear(10000, 10000),
            # nn.ReLU(),
            # nn.Linear(10000, 10000),
            # nn.ReLU(),
            nn.Linear(10000, 10000),
            # nn.BatchNorm1d(50000),
            nn.ReLU(),
            # nn.Linear(10000, 10000),
            # nn.ReLU(),
            nn.Linear(10000, 77 * 768)
        )
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = GroupLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True
    
    def forward(self, x):
        B, C, T = x.shape
        if T%200!=0: 
            x = x[:,:,0:T-T%200]
            T = T-T%200
        x = self.chan_conv(x)
        pred = self.feature(x)
        pred = self.head(pred)
        out = self.align(pred)
        return out,pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        eeg, text,image_emb,text_emb,depth,labels = batch
        eeg = eeg.float()
        text_embeddings = text.float()
        image_emb = image_emb.float()
        text_emb = text_emb.float()
        depth_emb = depth.float()
        x2, x1= self.forward(eeg)
        loss1 = F.mse_loss(x2, text_embeddings)
        loss3 = F.mse_loss(x1,text_emb)
        logit_scale = self.logit_scale
        img_loss = self.loss_func(x1, image_emb, logit_scale,labels)
        text_loss = self.loss_func(x1, text_emb, logit_scale,labels)
        depth_loss = self.loss_func(x1,depth_emb,logit_scale,labels)
        
        loss2 = (img_loss+text_loss+depth_loss)/3
        loss = 0.5*loss1 + 0.01*loss2 + loss3
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss
        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters())+
            list(self.feature.parameters()),
            weight_decay=0.01)#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
import datetime
import math
def get_time():
    current_time = datetime.datetime.now()

    # 加8小时并取mod 24
    new_hour = (current_time.hour + 8) % 24

    # 构建新的时间，替换小时部分
    new_time = current_time.replace(hour=new_hour)

    # 格式化输出
    formatted_time = new_time.strftime("%m-%d_%H-%M")
    return formatted_time
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
class NeuroclipDataset(torch.utils.data.Dataset):
    def __init__(self,eeg,text):
        self.eeg = eeg
        self.text = text 
    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, index):
        return self.eeg[index],self.text[index]
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

All_label = np.empty((0, 200))
for block_id in range(7):
    All_label = np.concatenate((All_label, GT_label[block_id].repeat(5).reshape(1, 200)))

eegdata = np.load('data/SEED-DV/Segmented_Rawf_200Hz_2s/sub1.npy')
# chosed_label = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]
chosed_label = [i for i in range(1,41)]
labels = np.zeros((40, 5, 62, 400))
for i in range(40):
    labels[i]=i+1
test_id = 6
train_eeg =[]
test_eeg = []
eeg=[]
EEG=[]
for i in range(6):
    indices = [list(GT_label[i]).index(element) for element in chosed_label]
    chosed_eeg = eegdata[i][indices,:]
    eeg.append(chosed_eeg)
    EEG.append(labels)
EEG = np.stack(EEG, axis=0)
eeg = np.stack(eeg,axis=0)
EEG = torch.from_numpy(EEG)
eeg = torch.from_numpy(eeg)
EEG = rearrange(EEG, 'a b c e f -> (a b c) (e f)')
eeg = rearrange(eeg, 'a b c e f -> (a b c) (e f)')
indices = [list(GT_label[6]).index(element) for element in chosed_label]
test_eeg =eegdata[6][indices,:]
test_eeg = rearrange(test_eeg,'a b c d -> (a b) (c d)')
normalize = StandardScaler()
normalize.fit(eeg)
EEG = normalize.transform(eeg) 
test_eeg = normalize.transform(test_eeg)
train_eeg = torch.from_numpy(EEG)
test_eeg = torch.from_numpy(test_eeg)
train_eeg = rearrange(train_eeg,'a (b c) -> a b c',c=400)
test_eeg = rearrange(test_eeg,'a (b c) -> a b c',c=400)

train_texts  = []
test_texts =[]

image_768 = torch.load('data/image_emb_768.pt',map_location='cpu')
text_768 = torch.load('data/text_emb_768.pt',map_location='cpu')
print(image_768.shape,text_768.shape)
image_768 = image_768[:,:,2]

Text = []
Text_768 = []
Image_768 = []
for i in range(6):
    text_embedding = torch.load(f'data/text_embeds/labels/block{i+1}.pt',map_location='cpu')
    text = rearrange(text_embedding,'(a b) c d -> a b c d',a=40)
    indices = [list(GT_label[i]).index(element) for element in chosed_label]
    text = text[indices,:][:,::5].repeat_interleave(5, dim=1)
    Text.append(text)

    text768 = rearrange(text_768[i],'(a b) d -> a b d',a=40)
    indices = [list(GT_label[i]).index(element) for element in chosed_label]
    text768 = text768[indices,:][:,::5].repeat_interleave(5, dim=1)

    img768 = rearrange(image_768[i],'(a b) d -> a b d',a=40)
    indices = [list(GT_label[i]).index(element) for element in chosed_label]
    img768 = img768[indices,:][:,::5].repeat_interleave(5, dim=1)
    Text_768.append(text768)
    Image_768.append(img768)
Text = torch.cat(Text,dim=0)
Text_768 = torch.cat(Text_768,dim=0)
Image_768 = torch.cat(Image_768,dim=0)
Text_768 = torch.reshape(Text_768,(-1,768))
Image_768 = torch.reshape(Image_768,(-1,768))
Text = torch.reshape(Text, (-1, Text.shape[2]*Text.shape[3]))
text_embedding = torch.load(f'data/text_embeds/labels/block7.pt',map_location='cpu')
test_texts = rearrange(text_embedding,'(a b) c d -> a b c d',a=40)
indices = [list(GT_label[6]).index(element) for element in chosed_label]
test_texts = test_texts[indices,:][:,::5].repeat_interleave(5, dim=1)
test_texts = torch.reshape(test_texts, (-1, 77*768))

print(train_eeg.shape,Text.shape)
print(test_eeg.shape,test_texts.shape)

batch_size=8

dataset = Dataset(train_eeg, Text,Image_768,Text_768)
test_dataset = NeuroclipDataset(test_eeg,  test_texts)
#valid_dataset = test_dataset
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
#valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

max_epochs = 200
steps_per_epoch = math.ceil(len(train_loader) )
max_lr = 1e-5
current_time ='05-12_15-55'
path='checkpoints/BIOT/05-12_15-55/best_model_40_classes-sub1.ckpt'
model = LitEEGPTCausal.load_from_checkpoint(path)


model.eval()
device = 'cuda'
model.to(device)
test_embed=[]
with torch.no_grad():
    for x,y in test_loader:
        x = x.to(device).float()
        y = y.to(device).float()
        output,_ = model(x)
        
        
        test_embed.append(output.cpu())



test_embed = torch.cat(test_embed,dim=0)

test_embed = test_embed.reshape(test_embed.shape[0],77,768)
print(test_embed.shape)

outdir = f'data/text_embeds/BIOT/{current_time}'
os.makedirs(outdir, exist_ok=True)
torch.save(test_embed, f'{outdir}/block7_40classes_1e-5-sub1.pt')