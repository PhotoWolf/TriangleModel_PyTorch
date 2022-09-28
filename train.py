import torch
import pandas as pd
import numpy as np
import tqdm.auto as tqdm

# +
def compute_phon_accuracy(preds,targets,embedding_matrix,k=2):
      ## Batch x 10 x 1 x 25
    preds = preds.view(preds.shape[0],-1,1,embedding_matrix.shape[-1])
    targets = targets.view(targets.shape[0],-1,1,embedding_matrix.shape[-1])

      ## Embedding matrix: 1 x 1 x 35 x 25
    pred_distances = (preds - embedding_matrix[None,None]).norm(dim=-1)
    target_distances = (targets - embedding_matrix[None,None]).norm(dim=-1)

    if k>1:        
        vals = (target_distances.argmin(dim=-1,keepdim=True) == pred_distances.argsort(dim=-1)[:,:,:k]).any(dim=-1)
    else:
        vals = (target_distances.argmin(dim=-1) == pred_distances.argmin(dim=-1))
    return vals.all(dim=-1).float()

def compute_semantic_accuracy(preds,targets,threshold=.5):
      return ((preds>threshold) == (targets>threshold)).all(dim=-1).float()

# -

def kl_divergence(targets,preds,eps=1e-6,zer=None):
    weighting = torch.ones(targets.shape[:-1],device=preds.device)
    weighting = weighting * torch.arange(1,preds.shape[0]+1,device=preds.device)[:,None]/preds.shape[0]
    weighting /= weighting.sum(dim=0,keepdim=True)

    loss = targets * torch.log(eps + targets/torch.clamp(preds,eps,1)) \
              + (1-targets) * torch.log(eps + (1-targets)/(1-torch.clamp(preds,0,1-eps)))

    if zer:
        mask = ((targets-preds).abs()>zer).float()

        loss = loss * mask
        loss = loss.sum(dim=-1)/(1e-6 + mask.sum(dim=-1))
    else:
        loss = loss.sum(dim=-1)

    return torch.sum(weighting * loss,dim=0).mean()

def get_metrics(ID,model,loader,device):
    phonology_accuracy = np.zeros(3)
    semantic_accuracy = np.zeros(3)
    for batch in loader:
        orthography,phonology,semantics = batch['orthography'].to(device),\
                                              batch['phonology'].to(device),\
                                              batch['semantics'].to(device)
        if 'phase_1' in ID:
           predicted_phonology,_ = model(0,4,1/3,semantics=semantics)
           _,predicted_semantics = model(0,4,1/3,phonology=phonology)
        elif 'phase_2' in ID:
           predicted_phonology,predicted_semantics = model(0,4,1/3,orthography=orthography)

        phonology_accuracy += np.array([compute_phon_accuracy(predicted_phonology[-1],phonology,
                                                torch.Tensor(loader.dataset.phonology_tokenizer.embedding_table.to_numpy()).to(device),
                                                k).sum().item() for k in [1,2,3]]
                                        )        
        semantic_accuracy += np.array([compute_semantic_accuracy(predicted_semantics[-1],semantics,threshold).sum().item() 
                                               for threshold in [.4,.5,.6]]
                                        )
    return phonology_accuracy/len(loader.dataset),semantic_accuracy/len(loader.dataset)

def train_loop(ID,model,opts,loader,task_prob,device,num_epochs=250,zer=None,lesions=[]):
    accuracy = []
    assert 'phase_1' in ID or 'phase_2' in ID

    with torch.no_grad():
        p_acc,s_acc = get_metrics(ID,model,loader,device)
    accuracy.append([p_acc,s_acc])

    losses = [[],[],[]]
    pbar = tqdm.tqdm(range(num_epochs),position=0)
    for epoch in pbar:
        if epoch%10 == 0:
           torch.save(model.state_dict(),f'ckpts/{ID.split("_")[-1]}_{epoch}')
        for idx,batch in enumerate(loader):
            orthography,phonology,semantics = batch['orthography'].to(device),\
                                                  batch['phonology'].to(device),\
                                                  batch['semantics'].to(device)

            task = np.random.choice([0,1,2],p=task_prob)
            if task == 0:
                predicted_phonology,predicted_semantics = model(2+2/3,4,1/3,phonology=phonology,semantics=semantics,lesions=lesions)
                phonology_loss = kl_divergence(phonology[None],predicted_phonology[-4::],zer=zer) 
                semantics_loss = kl_divergence(semantics[None],predicted_semantics[-4::],zer=zer)    
                
            if task == 1:
                _,predicted_semantics = model(0,4,1/3,phonology=phonology,lesions=lesions)
                predicted_phonology,_ = model(0,4,1/3,semantics=semantics,lesions=lesions)

                phonology_loss = kl_divergence(phonology[None],predicted_phonology[-3::],zer=zer) 
                semantics_loss = kl_divergence(semantics[None],predicted_semantics[-3::],zer=zer)    

            elif task == 2:
                predicted_phonology,predicted_semantics = model(0,4,1/3,orthography=orthography,lesions=lesions)
                
                phonology_loss = kl_divergence(phonology[None], predicted_phonology[2::],zer=zer)
                semantics_loss = kl_divergence(semantics[None], predicted_semantics[2::],zer=zer)

            loss = phonology_loss + semantics_loss
            loss.backward()
            opts[task].step()

            mean_grad,c = 0,0
            for name,parameter in model.named_parameters():
                if parameter.grad is not None:
                   mean_grad = parameter.grad.abs().sum() + c * mean_grad
                   c += parameter.numel()
                   mean_grad = mean_grad / c

            opts[task].zero_grad()
            losses[task].append([phonology_loss.item(),semantics_loss.item(),mean_grad.item()])

        with torch.no_grad():
            p_acc,s_acc = get_metrics(ID,model,loader,device)
            
        accuracy.append([p_acc,s_acc])
        torch.save(losses,f'{ID}_losses')
        torch.save(accuracy,f'{ID}_accuracy')
    return losses,accuracy
