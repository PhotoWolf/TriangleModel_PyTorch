import torch
from dataset import Monosyllabic_Dataset
from model import ModelConfig
from train import train_loop
import pickle


torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

dataset = Monosyllabic_Dataset('datasets/monosyllabic/df_train.csv',
                                  'datasets/phonetic_features.txt',
                                  'datasets/monosyllabic/sem_train.npz')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else: 
    device = torch.device('cpu')

loader = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=10,drop_last=True)

torch.manual_seed(0)
config = ModelConfig(orth_dim=110,phon_dim=250,sem_dim=2446)
model = config.create_model(lesions=['o2p']).to(device)

current_epoch = 0
if current_epoch:
   model.load_state_dict(torch.load(f'ckpts/no_o2p_{current_epoch-10}'))

opt = torch.optim.AdamW(model.parameters(),1e-3)
losses,accuracy = train_loop('baseline',model,opt,loader,device,num_epochs=500,
                                 current_epoch=current_epoch,zer=.1)
