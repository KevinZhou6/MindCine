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
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
import Modules.LaBraM.modeling_finetune

import timm.models
from timm.models import create_model
import torch




class LitEEGPTCausal(pl.LightningModule):

    def __init__(self):
        super().__init__() 
        
        checkpoint = torch.load("Modules/LaBraM/labram-base.pth")
        new_checkpoint = {}
        for k,v in checkpoint['model'].items():
            if k.startswith('student.'):
                new_checkpoint[k[len('student.'):]] = v
        model = create_model("labram_base_patch200_200", 
                                # checkpoint_path= ,
                                qkv_bias=False,
                                rel_pos_bias=True,
                                num_classes=4,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                attn_drop_rate=0.0,
                                drop_block_rate=None,
                                use_mean_pooling=True,
                                init_scale=0.001,
                                use_rel_pos_bias=True,
                                use_abs_pos_emb=True,
                                init_values=0.1,)
        model.load_state_dict(new_checkpoint, strict=False)
        for blk in model.blocks:
            for p in blk.parameters():
                p.requires_grad = False
        self.feature        = model
        self.head   =   LinearWithConstraint(25000, 77*768, max_norm=1)#
        self.loss_fn        = torch.nn.MSELoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True
        
    
    def forward(self, x):
        B, C, T = x.shape
        if T%200!=0: 
            x = x[:,:,0:T-T%200]
            T = T-T%200
        x = x.reshape((B,C,T//200,200))
        pred = self.feature.forward_features(x, input_chans=[i for i in range(C+1)], return_all_tokens=True)
        pred = self.head(pred.flatten(1))
        return x, pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x,y = batch
        x = x.float()
        y = y.float()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        
        return loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    # def on_validation_epoch_end(self) -> None:
    #     if self.is_sanity:
    #         self.is_sanity=False
    #         return super().on_validation_epoch_end()
            
    #     label, y_score = [], []
    #     for x,y in self.running_scores["valid"]:
    #         label.append(x)
    #         y_score.append(y)
    #     label = torch.cat(label, dim=0)
    #     y_score = torch.cat(y_score, dim=0)
    #     print(label.shape, y_score.shape)
        
        
    #     self.log('valid_loss', value, on_epoch=True, on_step=False, sync_dist=True)
    #     return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        
        x = x.float()
        y = y.float()
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
       
        
        #self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list(self.head.parameters())+
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
        
# load configs
# -- LOSO 

# load configs

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
seed_torch(9)

eegdata = np.load('/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/Models/EEG/EEG2Video/EEG2Video2/data/SEED-DV/Segmented_Rawf_200Hz_2s/sub1.npy')
chosed_label = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]
test_id = 6
train_eeg =[]
test_eeg = []
for i in range(7):
    indices = [list(GT_label[i]).index(element) for element in chosed_label]
    chosed_eeg = eegdata[i][indices,:]
    if i==test_id:
        test_eeg.append(chosed_eeg)
    else:
        train_eeg.append(chosed_eeg)
train_eeg = np.stack(train_eeg, axis=0)
test_eeg = np.stack(test_eeg, axis=0)
train_eeg = torch.from_numpy(train_eeg)
test_eeg = torch.from_numpy(test_eeg)

train_eeg = rearrange(train_eeg, 'a b c  e f -> (a b c)  (e f)') 
test_eeg = rearrange(test_eeg, 'a b c  e f -> (a b c)  (e f)') 
train_eeg =train_eeg.resize(train_eeg.shape[0], 62*400)
test_eeg = test_eeg.resize(test_eeg.shape[0], 62*400)
normalize = StandardScaler()
normalize.fit(train_eeg)
train_eeg = normalize.transform(train_eeg)  

test_eeg = normalize.transform(test_eeg)   
C=62
T=400
train_eeg = train_eeg.reshape(train_eeg.shape[0], C, T)
test_eeg = test_eeg.reshape(test_eeg.shape[0], C, T)
train_eeg = torch.from_numpy(train_eeg)
test_eeg = torch.from_numpy(test_eeg)


train_eeg = train_eeg
train_texts  = []
test_texts =[]

dic = []
for i in range(7):
    for j in range(10):
        texts = torch.load(f'../text_embeds/tune-a-video/block{i+1}.pt',map_location='cpu')
        #indices = [list(All_label[i]).index(element) for element in chosed_label]
        indices = np.isin(All_label[i], chosed_label[j])
        chose_texts = texts[indices][::5].repeat(5,1,1)
        
        
        if i ==test_id:
            test_texts.append(chose_texts if i==0 else dic[j])
            
        else:
            train_texts.append(chose_texts if i==0 else dic[j])
            
        if i==0:
            dic.append(chose_texts)
    

train_texts = torch.cat(train_texts,dim=0)
test_texts = torch.cat(test_texts,dim=0)
train_texts = train_texts.reshape(train_texts.shape[0],77*768)
test_texts = test_texts.reshape(test_texts.shape[0],77*768)
batch_size=4
train_dataset = NeuroclipDataset(train_eeg,train_texts)
test_dataset = NeuroclipDataset(test_eeg,  test_texts)
valid_dataset = test_dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

max_epochs = 200
steps_per_epoch = math.ceil(len(train_loader) )
max_lr = 5e-4
current_time =get_time()
# init model
model = LitEEGPTCausal()
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    dirpath=f'../checkpoints/Labram/{current_time}/',  # 模型保存的文件夹
    filename='best_model',  # 文件名
    monitor='val_loss',  # 用于监控的指标
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
                        logger=[pl_loggers.TensorBoardLogger('../Encoder_logs/', name="LaBraM", version="subject1"), 
                                pl_loggers.CSVLogger('../Encoder_logs/', name="LaBraM_csv")])

trainer.fit(model, train_loader, test_loader, ckpt_path='last')