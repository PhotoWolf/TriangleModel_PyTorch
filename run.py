import torch
import random
import pickle
import tqdm
import glob
import numpy as np
from dataset import Monosyllabic_Dataset
from model import ModelConfig
from train import train_loop

def fit(ID,model_config,loader,current_step,num_steps,lr=5e-3,lesions=[]):
   if torch.cuda.is_available():
       device = torch.device('cuda:0')
   else: 
       device = torch.device('cpu')

   torch.manual_seed(0)
   model = config.create_model(lesions=lesions).to(device)
   opt = torch.optim.Adam(model.parameters(),lr)

   if current_epoch:
      model.load_state_dict(torch.load(f'ckpts/{ID}_{current_epoch-10}'))

   losses,accuracy = train_loop(ID,model,opt,loader,device,num_steps=num_steps,
                                 current_step=current_step,zer=.1)

def step(ID,model,opt,loader,current_step,**kwargs):
   if torch.cuda.is_available():
       device = torch.device('cuda:0')
   else: 
       device = torch.device('cpu')
   losses,accuracy = train_loop(ID,model,opt,loader,device,num_steps=1,
                                 current_step=current_step,zer=.1,**kwargs)

   

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

dataset = Monosyllabic_Dataset('datasets/monosyllabic/df_train.csv',
                                  'datasets/phonetic_features.txt',
                                  'datasets/monosyllabic/sem_train.npz')


loader = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=10,drop_last=True)
config = ModelConfig(orth_dim=110,phon_dim=250,sem_dim=2446,learn_bias=False)

if torch.cuda.is_available():
   device = torch.device('cuda:0')
else: 
   device = torch.device('cpu')


model = config.create_model().to(device)

initial_step = 0
if initial_step:
   ckpt_path = glob.glob(f'ckpts/*_{initial_step}')[0]
   model.load_state_dict(torch.load(ckpt_path))

opt1 = torch.optim.AdamW(
                   list(model.cleanup['state_to_hidden']['semantics'].parameters()) +\
                   list(model.cleanup['hidden_to_state']['semantics'].parameters()),
                   5e-3,weight_decay=0)

opt2 = torch.optim.AdamW(
                   list(model.cleanup['state_to_hidden']['phonology'].parameters()) +\
                   list(model.cleanup['hidden_to_state']['phonology'].parameters()),
                   5e-3,weight_decay=0)

opt3 = torch.optim.AdamW(
                   list(model.cleanup['state_to_hidden']['semantics'].parameters()) +\
                   list(model.cleanup['hidden_to_state']['semantics'].parameters()) +\
                   list(model.phonology_semantics['state_to_hidden']['phonology'].parameters()) +\
                   list(model.phonology_semantics['hidden_to_state']['semantics'].parameters()),
                   5e-3,weight_decay=0)

opt4 = torch.optim.AdamW(
                   list(model.cleanup['state_to_hidden']['phonology'].parameters()) +\
                   list(model.cleanup['hidden_to_state']['phonology'].parameters()) +\
                   list(model.phonology_semantics['state_to_hidden']['semantics'].parameters()) +\
                   list(model.phonology_semantics['hidden_to_state']['phonology'].parameters()),
                   5e-3,weight_decay=0)

### Phase 1
for current_step in range(initial_step,0):

    if random.random() < .2:
       if random.random() < .5:
          model.lesions = ['o2s','o2p','p2s','s2p','p2p']
          step('phase_1_zeros_cleanup_s',model,opt1,loader,current_step,t_0=2 + 2/3,T=4,
                    start_error = -4,clamp_s = True)
       else:
          model.lesions = ['o2s','o2p','p2s','s2p','s2s']
          step('phase_1_zeros_cleanup_p',model,opt2,loader,current_step,t_0=2 + 2/3,T=4,
                    start_error = -4,clamp_p = True)

    else:
       if random.random() < .5:
          model.lesions = ['o2s','o2p','s2p','p2p']
          step('phase_1_zeros_p2s',model,opt3,loader,current_step,start_error = -3,T=4,clamp_p = True)
       else:
          model.lesions = ['o2s','o2p','p2s','s2s']
          step('phase_1_zeros_s2p',model,opt4,loader,current_step,start_error = -3,T=4,clamp_s = True)

    if (current_step+1)%(1e2) == 0:
        print("\n-------------------------------------------------\n")
        print(current_step)

        try:
           print(np.array(torch.load('metrics/phase_1_zeros_cleanup_s_accuracy'))[-int(1e2)::].mean(axis=0)[1])
        except:
           print(None)
        try:
           print(np.array(torch.load('metrics/phase_1_zeros_cleanup_p_accuracy'))[-int(1e2)::].mean(axis=0)[0])
        except:
           print(None)

        print(np.array(torch.load('metrics/phase_1_zeros_p2s_accuracy'))[-int(1e2)::].mean(axis=0)[1])
        print(np.array(torch.load('metrics/phase_1_zeros_s2p_accuracy'))[-int(1e2)::].mean(axis=0)[0])

### Phase 2
opt = torch.optim.AdamW(list(model.orthography_indirect.parameters())+
                       list(model.orthography_direct.parameters()),5e-3,weight_decay=0.0)
model.lesions = []
for current_step in range(51000):
    step('phase_2_zeros',model,opt,loader,current_step,start_error = 2)

    if (current_step+1)%(1e2) == 0:
        print("\n-------------------------------------------------\n")
        print(current_step)
        print(np.array(torch.load('metrics/phase_2_zeros_accuracy'))[-int(1e2)::].mean(axis=0))
