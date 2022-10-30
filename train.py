import torch
import pandas as pd
import numpy as np
import tqdm.auto as tqdm

BOUND = 15

def invert_binary(x):
    x[x==1] = BOUND
    x[x==0] = -BOUND
    return x

def create_inputs(model,batch_size,device,**kwargs):
    if kwargs.get('orthography',None) is not None:
       x_0 = {'orthography':invert_binary(kwargs['orthography'])}
    else:
       x_0 = {'orthography':-BOUND * torch.ones((batch_size,model.orth_dim),device=device)}
    
    if kwargs.get('phonology',None) is not None:
       x_0['phonology'] = invert_binary(kwargs['phonology'])
    else:
       x_0['phonology'] = -BOUND * torch.ones((batch_size,model.phon_dim),device=device)
              
    if kwargs.get('semantics',None) is not None:
       x_0['semantics'] = invert_binary(kwargs['semantics'])
    else:
       x_0['semantics'] = -BOUND * torch.ones((batch_size,model.sem_dim),device=device)
              
    x_0['cleanup_phon'] = -BOUND * torch.ones((batch_size,model.phon_cleanup_dim),device=device)
    x_0['cleanup_sem'] = -BOUND * torch.ones((batch_size,model.sem_cleanup_dim),device=device)

    x_0['sem_2_phon'] = -BOUND * torch.ones((batch_size,model.sem_2_phon_dim),device=device)
    x_0['phon_2_sem'] = -BOUND * torch.ones((batch_size,model.phon_2_sem_dim),device=device)

    x_0['orth_2_phon'] = -BOUND * torch.ones((batch_size,model.orth_2_phon_dim),device=device)
    x_0['orth_2_sem'] = -BOUND * torch.ones((batch_size,model.orth_2_sem_dim),device=device)
              
    return x_0

def forward_euler(f,x_0,t_0,T,delta_t):
    outputs,x = [x_0],x_0
    for t in torch.arange(t_0,T,delta_t):
        derivatives = f(x)
        for key in x:
            x[key] = x[key] + delta_t * derivatives[key]
            x[key] = torch.clamp(x[key],-BOUND,BOUND)
        outputs.append(x)
    return outputs

def collate_outputs(outputs):
    for idx,output in enumerate(outputs):
        S = torch.sigmoid(output['semantics'])[None]
        P = torch.sigmoid(output['phonology'])[None]

        if idx == 0:
           semantics = S
           phonology = P
        else:
           semantics = torch.cat((semantics,S),dim=0)
           phonology = torch.cat((phonology,P),dim=0)
    return phonology,semantics
# +
def compute_phon_accuracy(preds,targets,embedding_matrix,k=2):
    preds = preds.view(preds.shape[0],-1,1,embedding_matrix.shape[-1])
    targets = targets.view(targets.shape[0],-1,1,embedding_matrix.shape[-1])

    pred_distances = (preds - embedding_matrix[None,None]).norm(dim=-1)
    target_distances = (targets - embedding_matrix[None,None]).norm(dim=-1)

    vals = (target_distances.argmin(dim=-1,keepdim=True) == pred_distances.argsort(dim=-1)[:,:,:k]).any(dim=-1)
    return vals.all(dim=-1).float().mean()

def compute_sem_accuracy(preds,targets,threshold=.5):
    return ((preds>=threshold) == targets.bool()).all(dim=-1).float().mean()

def cross_entropy(preds,targets,zer,eps=1e-4):
    mask = ((targets-preds).abs()>=zer).float()

    cross_entropy = -targets * (eps + preds).log()
    cross_entropy = cross_entropy - (1-targets) * (1 + eps - preds).log()
    return (mask * cross_entropy).sum(dim=(-1,-2))/(eps + mask.sum(dim=(-1,-2)))

def train_loop(ID,model,opt,loader,device,num_epochs=250,current_epoch=0,zer=0.1):
    accuracy,losses = [],[]
    if current_epoch:
       accuracy = torch.load(f'metrics/{ID}_accuracy')
       losses = torch.load(f'metrics/{ID}_losses')

    pbar = tqdm.tqdm(range(current_epoch,current_epoch + num_epochs),position=0)
    for epoch in pbar:
        if epoch%10 == 0:
           torch.save(model.state_dict(),f'ckpts/{ID}_{epoch}')
           torch.save(opt.state_dict(),f'ckpts/{ID}_{epoch}_opt')
        for idx,batch in enumerate(loader):
            orthography,phonology,semantics = batch['orthography'].to(device),\
                                              batch['phonology'].to(device),\
                                              batch['semantics'].to(device)

            batch_size = orthography.shape[0]

            x_0 = create_inputs(model,batch_size,device,orthography=orthography)

            predicted_phonology,predicted_semantics = collate_outputs(
                                                          forward_euler(model,x_0,0,4,1/3)
                                                       )
            phonology_loss = cross_entropy(predicted_phonology[2::],phonology[None],zer)
            semantics_loss = cross_entropy(predicted_semantics[2::],semantics[None],zer)

            weighting = torch.arange(1,phonology_loss.shape[0]+1,device=device)
            weighting = weighting/weighting.sum()

            phonology_loss = (weighting * phonology_loss).sum()
            semantics_loss = (weighting * semantics_loss).sum()

            losses.append([phonology_loss.item(),semantics_loss.item()])
            loss = phonology_loss + semantics_loss

            loss.backward()
            opt.step()
            opt.zero_grad()

            embedding_table = torch.Tensor(loader.dataset.phonology_tokenizer.embedding_table.to_numpy())
            embedding_table = embedding_table.to(device)

            with torch.no_grad():
                p_acc = [compute_phon_accuracy(predicted_phonology[-1],phonology,embedding_table,k).item()
                        for k in [1,2,3]]
                s_acc = [compute_sem_accuracy(predicted_semantics[-1],semantics,t).item()
                        for t in [.4,.5,.6]]
             
            accuracy.append([p_acc,s_acc])
        print(np.mean(accuracy[-len(loader)::],axis=0),np.mean(losses[-len(loader)::],axis=0))
        torch.save(losses,f'metrics/{ID}_losses')
        torch.save(accuracy,f'metrics/{ID}_accuracy')

    return losses,accuracy
