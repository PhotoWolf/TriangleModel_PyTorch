import torch
import os
import argparse
import random
import pickle
import tqdm
import glob
import numpy as np
from dataset import Monosyllabic_Dataset
from model import ModelConfig
from train import Trainer,forward_euler,TrainerConfig

def run_p2p(trainer,model,opt,data):
    model.lesions = ['o2s','o2p','p2s','s2p','s2s']

    start_error = -4
    t_0 = 2 + 2/3

    inputs = {
                'phonology':data['phonology'].to(trainer.device),
                'semantics':data['semantics'].to(trainer.device),
              }
    
    targets = {
                'phonology':data['phonology'].to(trainer.device),
                'semantics':data['semantics'].to(trainer.device),
              }

    phon_loss,sem_loss,phon_acc,sem_acc = trainer.run(model,inputs,opt=opt,targets=targets)
    return phon_loss,phon_acc

def run_s2s(trainer,model,opt,data):
    model.lesions = ['o2s','o2p','p2s','s2p','p2p']

    start_error = -4
    t_0 = 2 + 2/3

    inputs = {
                'phonology':data['phonology'].to(trainer.device),
                'semantics':data['semantics'].to(trainer.device),
              }
    
    targets = {
                'phonology':data['phonology'].to(trainer.device),
                'semantics':data['semantics'].to(trainer.device),
              }

    phon_loss,sem_loss,phon_acc,sem_acc = trainer.run(model,inputs,opt=opt,targets=targets)
    return sem_loss,sem_acc

def run_sem_2_phon(trainer,model,opt,data):
    model.lesions = ['o2s','o2p','p2s','s2s']

    start_error = -3

    inputs = {'semantics':data['semantics'].to(trainer.device)}
    targets = {
                'phonology':data['phonology'].to(trainer.device),
                'semantics':data['semantics'].to(trainer.device),
              }
    loss,_,acc,_ = trainer.run(model,inputs,opt=opt,targets=targets)
    return loss,acc

def run_phon_2_sem(trainer,model,opt,data):
    model.lesions = ['o2s','o2p','s2p','p2p']

    start_error = -3

    inputs = {'phonology':data['phonology'].to(trainer.device)}
    targets = {
                'phonology':data['phonology'].to(trainer.device),
                'semantics':data['semantics'].to(trainer.device),
              }
    _,loss,_,acc = trainer.run(model,inputs,opt=opt,targets=targets)
    return loss,acc

def run_full(trainer,model,opt,data):
    model.lesions = []
    start_error = 2

    inputs = {'orthography':data['orthography'].to(trainer.device)}
    targets = {
                'phonology':data['phonology'].to(trainer.device),
                'semantics':data['semantics'].to(trainer.device),
              }
    phon_loss,sem_loss,phon_acc,sem_acc = trainer.run(model,inputs,opt=opt,targets=targets)
    return (phon_loss,sem_loss),(phon_acc,sem_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-ID',type=str,default='')
    
    parser.add_argument('-initial_step',type=int,default=0)
    parser.add_argument('-phase_1_steps',type=int,default=175000)
    parser.add_argument('-phase_2_steps',type=int,default=55000)
    
    parser.add_argument('-phase_1_eval_interval',type=int,default=5000)
    parser.add_argument('-phase_1_ckpt_interval',type=int,default=5000)

    parser.add_argument('-phase_2_eval_interval',type=int,default=1000)
    parser.add_argument('-phase_2_ckpt_interval',type=int,default=1000)
    
    args = parser.parse_args()
    
    os.makedirs('metrics',exist_ok=True)
    os.makedirs('ckpts',exist_ok=True)

    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)

    sample_dataset = Monosyllabic_Dataset('datasets/monosyllabic/df_train.csv',
                                   'datasets/phonetic_features.txt',
                                   'datasets/monosyllabic/sem_train.npz')
    no_sample_dataset = Monosyllabic_Dataset('datasets/monosyllabic/df_train.csv',
                                   'datasets/phonetic_features.txt',
                                   'datasets/monosyllabic/sem_train.npz',
                                   sample = False)

    sample_loader = torch.utils.data.DataLoader(sample_dataset,shuffle=True,batch_size=10,drop_last=True)
    no_sample_loader = torch.utils.data.DataLoader(no_sample_dataset,shuffle=True,batch_size=10,drop_last=False)
    phoneme_embeddings = torch.Tensor(sample_dataset.phonology_tokenizer.embedding_table.to_numpy())

    model_config = ModelConfig(orth_dim=110,phon_dim=250,sem_dim=2446,learn_bias=False)
    trainer_config = TrainerConfig()

    if torch.cuda.is_available():
       device = torch.device('cuda:0')
    else: 
       device = torch.device('cpu')

    trainer = trainer_config.create_trainer(phoneme_embeddings).to(device)
    model = model_config.create_model().to(device)

    if args.initial_step:
       ckpt_path = glob.glob(f'ckpts/*{args.ID}_{args.initial_step}')[0]
       model.load_state_dict(torch.load(ckpt_path))

    opt1 = torch.optim.AdamW(
                       list(model.cleanup['state_to_hidden']['phonology'].parameters()) +\
                       list(model.cleanup['hidden_to_state']['phonology'].parameters()),
                       5e-3,weight_decay=0
                       )

    opt2 = torch.optim.AdamW(
                       list(model.cleanup['state_to_hidden']['semantics'].parameters()) +\
                       list(model.cleanup['hidden_to_state']['semantics'].parameters()),
                       5e-3,weight_decay=0)

    opt3 = torch.optim.AdamW(
                       list(model.cleanup['state_to_hidden']['phonology'].parameters()) +\
                       list(model.cleanup['hidden_to_state']['phonology'].parameters()) +\
                       list(model.phonology_semantics['state_to_hidden']['semantics'].parameters()) +\
                       list(model.phonology_semantics['hidden_to_state']['phonology'].parameters()),
                       5e-3,weight_decay=0)

    opt4 = torch.optim.AdamW(
                       list(model.cleanup['state_to_hidden']['semantics'].parameters()) +\
                       list(model.cleanup['hidden_to_state']['semantics'].parameters()) +\
                       list(model.phonology_semantics['state_to_hidden']['phonology'].parameters()) +\
                       list(model.phonology_semantics['hidden_to_state']['semantics'].parameters()),
                       5e-3,weight_decay=0)

    ### Phase 1
    p2p_acc,p2p_loss = [],[]
    s2s_acc,s2s_loss = [],[]

    p2s_acc,p2s_loss = [],[]
    s2p_acc,s2p_loss = [],[]

    eval_p2s_acc = []
    eval_s2p_acc = []

    last_val = True    
    for current_step in range(args.initial_step,args.phase_1_steps):
        for data in sample_loader:
            break;

        if random.random() < .2:
           if random.random() < .5:
              losses,accs = run_p2p(trainer,model,opt1,data)

              p2p_loss.append(losses.item())
              p2p_acc.append([a.item() for a in accs])

              np.save(f'metrics/{args.ID}_train_p2p_loss',p2p_loss)
              np.save(f'metrics/{args.ID}_train_p2p_acc',p2p_acc)

           else:
              losses,accs = run_s2s(trainer,model,opt2,data)

              s2s_loss.append(losses.item())
              s2s_acc.append([a.item() for a in accs])

              np.save(f'metrics/{args.ID}_train_s2s_loss',s2s_loss)
              np.save(f'metrics/{args.ID}_train_s2s_acc',s2s_acc)

        else:
           if random.random() < .5:
              loss,acc = run_sem_2_phon(trainer,model,opt3,data)

              s2p_loss.append(loss.item())
              s2p_acc.append([a.item() for a in acc])

              np.save(f'metrics/{args.ID}_train_s2p_loss',s2p_loss)
              np.save(f'metrics/{args.ID}_train_s2p_acc',s2p_acc)

           else:
              loss,acc = run_phon_2_sem(trainer,model,opt4,data)

              p2s_loss.append(loss.item())
              p2s_acc.append([a.item() for a in acc])

              np.save(f'metrics/{args.ID}_train_p2s_loss',p2s_loss)
              np.save(f'metrics/{args.ID}_train_p2s_acc',p2s_acc)

        if (current_step+1)%(1e2) == 0:

            if last_val: 
               print("\n---------------------Train----------------------------\n")
               last_val = False
            else:
               print("\n------------------------------------------------------\n")

            print(current_step)

            try:
               print(np.array(np.load(f'metrics/{args.ID}_train_p2p_acc.npy'))[-int(1e2)::].mean(axis=0))
            except:
               print(None)
            try:
               print(np.array(np.load(f'metrics/{args.ID}_train_s2s_acc.npy'))[-int(1e2)::].mean(axis=0))
            except:
               print(None)

            print(np.array(np.load(f'metrics/{args.ID}_train_s2p_acc.npy'))[-int(1e2)::].mean(axis=0))
            print(np.array(np.load(f'metrics/{args.ID}_train_p2s_acc.npy'))[-int(1e2)::].mean(axis=0))
            
        if (current_step+1)%(args.phase_1_ckpt_interval) == 0:
            torch.save(model.state_dict(),f'ckpts/phase_1_{args.ID}_{current_step}.pth')
            
        if (current_step+1)%(args.phase_1_eval_interval) == 0:
            all_s2p_acc,all_p2s_acc = 3 * [0], 3 * [0]
            for data in no_sample_loader:
                _,acc = run_sem_2_phon(trainer,model,None,data)

                for jdx,a in enumerate(acc):
                    all_s2p_acc[jdx] += data['phonology'].shape[0] * a

                _,acc = run_phon_2_sem(trainer,model,None,data)

                for jdx,a in enumerate(acc):
                    all_p2s_acc[jdx] += data['phonology'].shape[0] * a
                
            eval_s2p_acc.append([a.item()/len(no_sample_dataset) for a in all_s2p_acc])
            eval_p2s_acc.append([a.item()/len(no_sample_dataset) for a in all_p2s_acc])

            print("\n--------------------Eval-----------------------------\n")
            print(current_step)
            print(eval_s2p_acc[-1])
            print(eval_p2s_acc[-1])

            last_val = True
            
            np.save(f'metrics/{args.ID}_eval_s2p_acc',eval_s2p_acc)
            np.save(f'metrics/{args.ID}_eval_p2s_acc',eval_p2s_acc)
            
    ### Phase 2
    opt = torch.optim.AdamW(
                          list(model.orthography_indirect.parameters())+
                          list(model.orthography_direct.parameters()),
                      5e-3,weight_decay=0.0)

    o2p_loss,o2p_acc = [],[]
    o2s_loss,o2s_acc = [],[]
    
    eval_o2p_acc = []
    eval_o2s_acc = []

    last_val = True
    for current_step in range(args.phase_2_steps):

        for data in sample_loader:
            break;

        losses,accs = run_full(trainer,model,opt,data)

        np.save(f'metrics/{args.ID}_train_o2p_loss',losses[0])
        np.save(f'metrics/{args.ID}_train_o2p_acc',accs[0])

        np.save(f'metrics/{args.ID}_train_o2s_loss',losses[1])
        np.save(f'metrics/{args.ID}_train_o2s_acc',accs[1])

        if (current_step+1)%(1e2) == 0:
            if last_val:
                print("\n---------------------Train----------------------------\n")
                last_val = False
            else:
                print("\n------------------------------------------------------\n")

            print(current_step)

            print(np.array(np.load(f'metrics/{args.ID}_train_o2p_acc.npy'))[-int(1e2)::].mean(axis=0))
            print(np.array(np.load(f'metrics/{args.ID}_train_o2s_acc.npy'))[-int(1e2)::].mean(axis=0))
            
        if (current_step+1)%(args.phase_2_ckpt_interval) == 0:
            torch.save(model.state_dict(),f'ckpts/phase_2_{args.ID}_{current_step}.pth')
            
        if (current_step+1)%(args.phase_2_eval_interval) == 0:
            all_o2p_acc,all_o2s_acc = 0,0
            for data in no_sample_loader:
                _,_,acc1,acc2 = run_full(trainer,model,None,data)
                
                for jdx,a in enumerate(acc1):
                   all_o2p_acc[jdx] += data['phonology'].shape[0] * a
                   all_o2s_acc[jdx] += data['semantics'].shape[0] * acc2[jdx]

            eval_o2p_acc.append([a.item()/len(no_sample_dataset) for a in all_o2p_acc])
            eval_o2s_acc.append([a.item()/len(no_sample_dataset) for a in all_o2s_acc])
                
            print("\n--------------------Eval-----------------------------\n")
            print(current_step)
            print(eval_o2p_acc[-1])
            print(eval_o2s_acc[-1])

            last_val = True            

            np.save(f'metrics/{args.ID}_eval_o2p_acc',eval_o2p_acc)
            np.save(f'metrics/{args.ID}_eval_o2s_acc',eval_o2s_acc)
