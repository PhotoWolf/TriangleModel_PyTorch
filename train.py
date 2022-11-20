import torch
import pandas as pd
import numpy as np
import tqdm.auto as tqdm

BOUND = 15

def invert_binary(tensor):
    new_tensor = BOUND * torch.ones_like(tensor,device=tensor.device)
    new_tensor[tensor == 0] = -BOUND
    return new_tensor

def forward_euler(f,x_0,t_0,T,delta_t):
    outputs,x = [x_0],x_0
    for t in torch.arange(0,T,delta_t):
        derivatives = f(x)
        nx = {}
        for key in x:
            if t<t_0 and key in ['phonology','semantics']:
               nx[key] = x[key]
            nx[key] = x[key] + delta_t * derivatives[key]
            nx[key] = torch.clamp(nx[key],-BOUND,BOUND)
        outputs.append(nx)
        x = nx
    return outputs

class Metrics:
    def __init__(self,phoneme_embedding_matrix,k=[1],tau=[.5]):
        self.phoneme_embedding_matrix = phoneme_embedding_matrix
        self.k = k
        self.tau = tau
  
    def to(self,device):
        assert isinstance(device,torch.device)
        self.phoneme_embedding_matrix = self.phoneme_embedding_matrix.to(device)
        return self

    def compute_phon_accuracy(self,preds,targets,k):
        preds = preds.view(preds.shape[0],-1,1,self.phoneme_embedding_matrix.shape[-1])
        targets = targets.view(targets.shape[0],-1,1,self.phoneme_embedding_matrix.shape[-1])

        pred_distances = (preds - self.phoneme_embedding_matrix[None,None]).norm(dim=-1)
        target_distances = (targets - self.phoneme_embedding_matrix[None,None]).norm(dim=-1)

        vals = (target_distances.argmin(dim=-1,keepdim=True) == pred_distances.argsort(dim=-1)[:,:,:k]).any(dim=-1)
        return vals.all(dim=-1).float().mean()


    def compute_sem_accuracy(self,preds,targets,tau):
        return ((preds>=tau) == targets.bool()).all(dim=-1).float().mean()

    def __call__(self,preds,targets):
        phonology_preds = preds['phonology']
        phonology_targets = targets['phonology']

        semantics_preds = preds['semantics']
        semantics_targets = targets['semantics']

        phon_accuracy = [self.compute_phon_accuracy(phonology_preds,phonology_targets,k) for k in self.k]
        sem_accuracy = [self.compute_sem_accuracy(semantics_preds,semantics_targets,tau) for tau  in self.tau]
        return phon_accuracy,sem_accuracy

class TrainerConfig:
    def __init__(self,**kwargs):
        self.params = kwargs
        
    @classmethod
    def from_json(cls,json_path):
        config_params = json.load(open(json_path,'r'))
        return cls(**config_params)
        
    def create_trainer(self,phoneme_embedding_matrix):
        if self.params.get('solver','forward_euler') == 'forward_euler':
            solver = forward_euler
        else:
            raise ValueError('Supported solvers include: forward_euler')
        return Trainer(solver,phoneme_embedding_matrix,self.params.get('zer',.1))
    
class Trainer:
    def __init__(self,solver,phoneme_embedding_matrix,zer):

        self.solver = solver
        self.metrics = Metrics(phoneme_embedding_matrix,[1,2,3],[.4,.5,.6])
        self.zer = zer
        self.device = torch.device('cpu')
        
    def to(self,device):
        assert isinstance(device,torch.device)
        self.device = device
        self.metrics.to(device)
        return self

    def cross_entropy(self,preds,targets,zer,eps=1e-4):
        mask = ((targets-preds).abs()>=zer).float()

        cross_entropy = -targets * (eps + preds).log()
        cross_entropy = cross_entropy - (1-targets) * (1 + eps - preds).log()
        return (mask * cross_entropy).sum(dim=(-1,-2))/(eps + mask.sum(dim=(-1,-2)))

    def collate_outputs(self,outputs):
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
 
    def create_inputs(self,model,data):
       temp = torch.zeros((0,))
#       print(data)
       batch_size = max([
                        data.get('orthography',temp).shape[0],
                        data.get('phonology',temp).shape[0],
                        data.get('semantics',temp).shape[0],
                    ])

       if data.get('orthography',None) is not None:
          inputs = {'orthography':invert_binary(data['orthography'])}
       else:
          inputs = {'orthography':-BOUND * torch.ones((batch_size,model.orth_dim),device=self.device)}

       if data.get('phonology',None) is not None:
          inputs['phonology'] = invert_binary(data['phonology'])
       else:
          inputs['phonology'] = -BOUND * torch.ones((batch_size,model.phon_dim),device=self.device)

       if data.get('semantics',None) is not None:
          inputs['semantics'] = invert_binary(data['semantics'])
       else:
          inputs['semantics'] = -BOUND * torch.ones((batch_size,model.sem_dim),device=self.device)

       inputs['cleanup_phon'] = -BOUND * torch.ones((batch_size,model.phon_cleanup_dim),device=self.device)
       inputs['cleanup_sem'] = -BOUND * torch.ones((batch_size,model.sem_cleanup_dim),device=self.device)

       inputs['sem_2_phon'] = -BOUND * torch.ones((batch_size,model.sem_2_phon_dim),device=self.device)
       inputs['phon_2_sem'] = -BOUND * torch.ones((batch_size,model.phon_2_sem_dim),device=self.device)

       inputs['orth_2_phon'] = -BOUND * torch.ones((batch_size,model.orth_2_phon_dim),device=self.device)
       inputs['orth_2_sem'] = -BOUND * torch.ones((batch_size,model.orth_2_sem_dim),device=self.device)

       return inputs

    def run(self,model,inputs,targets=None,opt=None,**kwargs):

        start_error = kwargs.get('start_error',2)
        delta_t = kwargs.get('delta_t',1/3)
        t_0 = kwargs.get('t_0',0)
        T = kwargs.get('T',4)

        inputs = self.create_inputs(model,inputs)
        outputs = self.solver(model,inputs,t_0,T,delta_t)
        
        predicted_phonology,predicted_semantics = self.collate_outputs(outputs)

        if targets is None:
           if kwargs.get('return_outputs',False):
              return outputs
           else:
              return predicted_phonology,predicted_semantics

        else: 
            phonology = targets['phonology']
            semantics = targets['semantics']
 
            p_acc,s_acc = self.metrics(
                          {'phonology':predicted_phonology[-1],'semantics':predicted_semantics[-1]},
                          targets)

            if opt is None:
               return None,None,p_acc,s_acc

            else:
               phonology_loss = self.cross_entropy(predicted_phonology[start_error::],phonology[None],self.zer)
               semantics_loss = self.cross_entropy(predicted_semantics[start_error::],semantics[None],self.zer)

               weighting = torch.arange(1,phonology_loss.shape[0]+1,device=self.device)
               weighting = weighting/weighting[-1]

               phonology_loss = (weighting * phonology_loss).sum()
               semantics_loss = (weighting * semantics_loss).sum()

               loss = phonology_loss + semantics_loss
               loss.backward()

               opt.step()
               opt.zero_grad()

               return phonology_loss,semantics_loss,p_acc,s_acc
