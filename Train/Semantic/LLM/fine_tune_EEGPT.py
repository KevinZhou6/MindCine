import random 
import os
import torch
from torch import nn
import pytorch_lightning as pl  
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from functools import partial
import numpy as np
import random
import os 
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
from einops import rearrange
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
seed_torch(200212299)
from Modules.EEGPT.EEGPT_mcae import EEGTransformer

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint

import timm.models
from timm.models import create_model
import torch
from loss import ClipLoss,soft_clip_loss,GroupLoss
import torch.nn.functional as F
use_channels_names =['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4',
            'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3',
            'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7',
            'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
            'CB1', 'O1', 'Oz', 'O2', 'CB2']
class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, load_path="/userhome2/zhoutianyi/Foundation Model/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()    
        self.chans_num = 62
        # init model
        target_encoder = EEGTransformer(
            img_size=[62, 400],
            patch_size=32*2,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        self.target_encoder = target_encoder
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path,weights_only=False)
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
                
        self.target_encoder.load_state_dict(target_encoder_stat)
        # self.chan_conv       = Conv1dWithConstraint(22, self.chans_num, 1, max_norm=1)
        self.feature=self.target_encoder
        self.head = nn.Sequential(
            nn.Linear(2048*6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
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
        self.loss_fn        = torch.nn.MSELoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = GroupLoss() 
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True
        self.logits = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, x):
        
        # x = self.chan_conv(x)
        
        self.target_encoder.eval()
        
        z = self.target_encoder(x, self.chans_id.to(x))
        
        pred = self.head(z.flatten(1)) #[b,2048*6]
        out = self.align(pred)
        return out, pred
        
        

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
        img_loss = model.loss_func(x1, image_emb, logit_scale,labels)
        text_loss = model.loss_func(x1, text_emb, logit_scale,labels)
        depth_loss = model.loss_func(x1,depth_emb,logit_scale,labels)
        
        loss2 = (img_loss+text_loss+depth_loss)/3
        loss = 0.5*loss1 + 0.01*loss2 + loss3
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        
        
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list(self.head.parameters())+
            
            list(self.align.parameters()),
            )#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'train_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        
# load configs
# -- LOSO 

# load configs
import datetime
import math
def get_time():
    current_time = datetime.datetime.now()

    # 加8小时并取mod 24
    new_hour = (current_time.hour) % 24

    # 构建新的时间，替换小时部分
    new_time = current_time.replace(hour=new_hour)

    # 格式化输出
    formatted_time = new_time.strftime("%m-%d_%H-%M")
    return formatted_time

class Dataset():
    def __init__(self, eeg, text,img_emb,text_emb,depth,labels):

        # scaler = preprocessing.StandardScaler().fit(eeg)
        # eeg = scaler.transform(eeg)

        self.eeg = eeg
        self.text = text
        self.img_emb = img_emb
        self.text_emb = text_emb
        self.depth =depth
        self.labels = labels
        self.len = eeg.shape[0]


    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.text[item], self.img_emb[item], self.text_emb[item],self.depth[item],self.labels[item]


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
eegdata = np.load('data/SEED-DV/Segmented_Rawf_200Hz_2s/sub3.npy')
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
test_eeg = labels
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
text_768 = torch.load('/data/text_emb_768.pt',map_location='cpu')
depth = torch.load('data/depth_768.pt',map_location='cpu')
print(image_768.shape,text_768.shape)
image_768 = image_768[:,:,2]
Text = []
Text_768 = []
Image_768 = []
Depth_768 =[]
Train_labels=[]
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

    depth768 = rearrange(depth[i],'(a b) d -> a b d',a=40)
    indices = [list(GT_label[i]).index(element) for element in chosed_label]
    depth768 = depth768[indices,:][:,::5].repeat_interleave(5, dim=1)

    train_labels = torch.from_numpy(np.array(chosed_label))
    train_labels = train_labels.repeat_interleave(5)

    Text_768.append(text768)
    Image_768.append(img768)
    Depth_768.append(depth768)
    Train_labels.append(train_labels)
Text = torch.cat(Text,dim=0)
Text_768 = torch.cat(Text_768,dim=0)
Image_768 = torch.cat(Image_768,dim=0)
Text_768 = torch.reshape(Text_768,(-1,768))
Image_768 = torch.reshape(Image_768,(-1,768))
Depth_768 = torch.cat(Depth_768,dim=0)
Depth_768 = torch.reshape(Depth_768,(-1,768))
Train_labels = torch.cat(Train_labels,dim=0)
Text = torch.reshape(Text, (-1, Text.shape[2]*Text.shape[3]))
# test_texts = rearrange(text_embedding,'(a b) c d -> a b c d',a=40)
# indices = [list(GT_label[6]).index(element) for element in chosed_label]
# test_texts = test_texts[indices,:][:,::5].repeat_interleave(5, dim=1)
# test_texts = torch.reshape(test_texts, (-1, 77*768))

print(train_eeg.shape,Text.shape)
# print(test_eeg.shape,test_texts.shape)

batch_size=8

dataset = Dataset(train_eeg, Text,Image_768,Text_768,Depth_768,Train_labels)
# test_dataset = NeuroclipDataset(test_eeg,  test_texts)
#valid_dataset = test_dataset
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)

max_epochs = 200
steps_per_epoch = math.ceil(len(train_loader) )
max_lr = 1e-5
current_time =get_time()
#path='/userhome2/zhoutianyi/EEG2Video/checkpoints/EEGPT_fake/04-26_00-25/best_model_20classes_5e-6_200212299.ckpt'
#model = LitEEGPTCausal.load_from_checkpoint(path)
model = LitEEGPTCausal()
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    dirpath=f'../checkpoints/EEGPT/{current_time}/',  # 模型保存的文件夹
    filename='best_model_40classes_1e-5-sub3',  # 文件名
    monitor='train_loss',  # 用于监控的指标
    mode='min',  # 'min'表示选取最小值（适用于loss类的指标）
    save_top_k=1,  # 只保存最佳的一个模型
    save_weights_only=True  # 只保存权重
)
# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [checkpoint_callback,lr_monitor]

trainer = pl.Trainer(accelerator='cuda',
                        
                        max_epochs=max_epochs, 
                        callbacks=callbacks,
                        logger=[ 
                                pl_loggers.CSVLogger('../logs/', name="EEGPT_csv",version=f'{current_time}')],
                    strategy='ddp_find_unused_parameters_true')

trainer.fit(model, train_loader, ckpt_path='last')
