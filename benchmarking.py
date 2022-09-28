import torch
from dataset import Monosyllabic_Dataset
from model import ModelConfig,st_clamp
from train import train_loop
import pickle

dataset = Monosyllabic_Dataset('df_train.csv','phonetic_features.txt','sem_train.npz')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else: 
    device = torch.device('cpu')

loader = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=10)

config = ModelConfig(orth_dim=110,phon_dim=250,sem_dim=2446, init_states=False)
model = config.create_model([lambda x: x, lambda x: x]).to(device)

opts = [torch.optim.AdamW(list(model.cleanup['state_to_hidden'].parameters())+
                           list(model.cleanup['hidden_to_state'].parameters())+
                           list(model.default_hidden['cleanup'].parameters()),1e-2),
        torch.optim.AdamW(list(model.phonology_semantics['state_to_hidden'].parameters())
                            +list(model.phonology_semantics['hidden_to_state'].parameters())
                            +list(model.default_hidden['phonology_semantics'].parameters())
                            +list(model.default_inputs.parameters())
                            +list(model.cleanup['state_to_hidden'].parameters())
                            +list(model.cleanup['hidden_to_state'].parameters())
                            +list(model.default_hidden['cleanup'].parameters()),1e-2),
        torch.optim.AdamW(list(model.orthography_indirect['state_to_hidden'].parameters())
                            +list(model.orthography_indirect['hidden_to_state'].parameters())
                            +list(model.orthography_direct['phonology'].parameters())
                            +list(model.default_hidden['orthography'].parameters()),1e-2)]

losses,accuracy = train_loop('phase_1_phon',model,opts,loader,[1,0,0],device,num_epochs=300,zer=.1,lesions=['o2s','p2s','s2p'])
losses,accuracy = train_loop('phase_2_phon',model,opts,loader,[0,0,1],device,num_epochs=300,zer=.1,lesions=['o2s','p2s','s2p'])
