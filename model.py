import torch
import json
from typing import Optional,List,Dict
from dataclasses import dataclass

class TimeAveragedInputs(torch.nn.Module):
   def __init__(self,in_features : int, out_features : int,bias : Optional[bool] =False):
       '''
        Time-Averaged Inputs formulation of the gradient (refer to
        Plaut et. al, 1998). Definition is slightly modified to 
        account for lesioning.

        Args:
           in_features (int) : dimemsionality of [X]
           out_features (int) : dimensionality of [Y]
           bias (bool) : if True, use trainable bias term

       '''
       super(TimeAveragedInputs,self).__init__()
       self.W1 = torch.nn.Linear(in_features,out_features,bias=bias)

   def forward(self, X : torch.Tensor, Y : torch.Tensor, s : int) -> torch.Tensor:
       '''
         Compute the contribution of [X] to the gradient of [Y] w.r.t to time. 
         
         Args:
            X (torch.Tensor) : dimensionality [in_features]
            Y (torch.Tensor) : dimensionality [out_features]
            s (int) : scales the -Y term according to the 
                      # of participating lesions.
         Returns:
            gradient contribution; dimensionality [out_features]
       '''
       return self.W1(torch.sigmoid(X)) - (1/s) * Y

@dataclass()
class ModelConfig:
    '''
       Configuration class to define TriangleModel architecture.
       
       Args:
           learn_bias (bool) : whether to include bias term in gradient computation.
           
           orth_dim (int) : dimensionality of the orthographic state
           phon_dim (int) : dimensionality of the phonological state
           sem_dim (int) : dimensionality of the semantic state
           
           phon_cleanup_dim (int) : dimensionality of the phonological cleanup unit
           sem_cleanup_dim (int) : dimensionality of the semantic cleanup unit
           
           phon_2_sem_dim (int) : dimensionality of the phonology -> semantics 
                                  hidden unit
           sem_2_phon_dim (int) : dimensionality of the semantics -> phonology
                                  hidden unit
                                 
           orth_2_sem_dim (int) : dimensionality of the orthography -> semantics
                                  hidden unit.
           orth_2_phon_dim (int) : dimensionality of the orthography -> phonology
                                   hidden unit.
    '''
    
    learn_bias : bool = False

    orth_dim : int = 111
    phon_dim : int = 200
    sem_dim : int = 1989

    phon_cleanup_dim : int = 50
    sem_cleanup_dim : int = 50

    phon_2_sem_dim : int = 500
    sem_2_phon_dim : int = 500

    orth_2_sem_dim : int = 500
    orth_2_phon_dim : int = 100

    @classmethod
    def from_json(cls,json_path : str) -> 'ModelConfig':
        '''
          Read config parameters from .json file
          
          Args:
             json_path (str) : path to config file
          Return:
             ModelConfig
        '''
        config_params = json.load(open(json_path,'r'))
        return cls(**config_params)
        
    def create_model(self,operator : Optional[torch.nn.Module] = TimeAveragedInputs,
                          lesions : Optional[List[str]] = []) -> 'TriangleModel':
        '''
          Instantiate TriangleModel w/ desired parameters
          
          Args:
              operator (torch.nn.Module): module to compute gradient contributions.
                                          Defaults to TimeAveragedInputs.
              lesions (List[str]): list of lesions to apply to the model. Accepts
                                   values of 'p2p', 's2s', 'p2s', 's2p', 'o2s', 
                                   and 'o2p'.
              
          Return:
              TriangleModel 
        '''
        return TriangleModel(self.orth_dim,self.phon_dim,self.sem_dim,
                                self.phon_cleanup_dim,self.sem_cleanup_dim,
                                self.phon_2_sem_dim,self.sem_2_phon_dim,
                                self.orth_2_sem_dim,self.orth_2_phon_dim,
                                bool(self.learn_bias),operator,lesions)

class TriangleModel(torch.nn.Module):
    def __init__(self, orth_dim : int, phon_dim : int, sem_dim : int,
                    phon_cleanup_dim : int, sem_cleanup_dim : int,
                    phon_2_sem_dim : int, sem_2_phon_dim : int,
                    orth_2_sem_dim : int, orth_2_phon_dim : int,
                    learn_bias : bool, operator : torch.nn.Module,
                    lesions : List[str]):
        super(TriangleModel,self).__init__()
        '''
          A PyTorch implemtation of the Triangle Model detailed in
          Harm and Seidenberg, 2004.
          
          Args:           
              orth_dim (int) : dimensionality of the orthographic state
              phon_dim (int) : dimensionality of the phonological state
              sem_dim (int) : dimensionality of the semantic state

              phon_cleanup_dim (int) : dimensionality of the phonological cleanup unit
              sem_cleanup_dim (int) : dimensionality of the semantic cleanup unit

              phon_2_sem_dim (int) : dimensionality of the phonology -> semantics 
                                     hidden unit
              sem_2_phon_dim (int) : dimensionality of the semantics -> phonology
                                     hidden unit

              orth_2_sem_dim (int) : dimensionality of the orthography -> semantics
                                     hidden unit.
              orth_2_phon_dim (int) : dimensionality of the orthography -> phonology
                                      hidden unit.
                                      
              learn_bias (bool) : whether to include bias term in gradient computation.
              
              operator (torch.nn.Module): module to compute gradient contributions.
                                          Defaults to TimeAveragedInputs.
                                          
              lesions (List[str]): list of lesions to apply to the model. Accepts
                                   values of 'p2p', 's2s', 'p2s', 's2p', 'o2s', 
                                   and 'o2p'.

        '''
        self.lesions = lesions

        self.orth_dim,self.phon_dim,self.sem_dim = orth_dim,phon_dim,sem_dim
        self.phon_cleanup_dim,self.sem_cleanup_dim = phon_cleanup_dim,sem_cleanup_dim
        self.phon_2_sem_dim,self.sem_2_phon_dim = phon_2_sem_dim,sem_2_phon_dim
        self.orth_2_sem_dim,self.orth_2_phon_dim = orth_2_sem_dim,orth_2_phon_dim

        ### Instantiate Cleanup Units
        self.cleanup = torch.nn.ModuleDict({'state_to_hidden':torch.nn.ModuleDict({
                                        'semantics':operator(sem_dim,sem_cleanup_dim,learn_bias),
                                        'phonology':operator(phon_dim,phon_cleanup_dim,learn_bias)}),
                                            'hidden_to_state':torch.nn.ModuleDict({
                                        'semantics':operator(sem_cleanup_dim,sem_dim,learn_bias),
                                        'phonology':operator(phon_cleanup_dim,phon_dim,learn_bias)})
                                        })
        
        ### Instantiate S2P and P2S pathways
        self.phonology_semantics = torch.nn.ModuleDict({'state_to_hidden':torch.nn.ModuleDict({
                                                 'semantics':operator(sem_dim,sem_2_phon_dim,learn_bias),
                                                 'phonology':operator(phon_dim,phon_2_sem_dim,learn_bias)}),
                                                        'hidden_to_state':torch.nn.ModuleDict({
                                                 'phonology':operator(sem_2_phon_dim,phon_dim,learn_bias),
                                                 'semantics':operator(phon_2_sem_dim,sem_dim,learn_bias)})
                                                 })
        
        ### Instantiate indirect O2P and O2S pathways
        self.orthography_indirect = torch.nn.ModuleDict({'state_to_hidden':torch.nn.ModuleDict({
                                                  'semantics':operator(orth_dim,orth_2_sem_dim,learn_bias), 
                                                  'phonology':operator(orth_dim,orth_2_phon_dim,learn_bias)}),
                                                        'hidden_to_state':torch.nn.ModuleDict({
                                                  'semantics':operator(orth_2_sem_dim,sem_dim,learn_bias),
                                                  'phonology':operator(orth_2_phon_dim,phon_dim,learn_bias)})
                                                  })
        
        ### Instantiate direct O2P and O2S pathways
        self.orthography_direct = torch.nn.ModuleDict({
                                                   'semantics':operator(orth_dim,sem_dim,learn_bias),
                                                   'phonology':operator(orth_dim,phon_dim,learn_bias)}
                                                   )
    def forward(self,inputs : Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
        '''
           Compute gradients of all states / hidden units w.r.t to time.
           
           Args:
              inputs (Dict[str,torch.Tensor]) : Values of all states / hidden units
                                                at the current timestep.
           Returns:
              Gradients of all states / hidden units
        '''
        
        ### Get states
        orthography = inputs['orthography']
        phonology = inputs['phonology']
        semantics = inputs['semantics'] 

        ### Get cleanup units
        cleanup_phon = inputs['cleanup_phon']
        cleanup_sem = inputs['cleanup_sem']

        ### Get oral hidden units
        phon_2_sem = inputs['phon_2_sem']
        sem_2_phon = inputs['sem_2_phon']

        ### Get reading hidden units
        orth_2_sem = inputs['orth_2_sem']
        orth_2_phon = inputs['orth_2_phon']

        ### Get lesions
        if 'o2p' in self.lesions:
           o2p_lesion = 0
        else:
           o2p_lesion = 1

        if 'o2s' in self.lesions:
           o2s_lesion = 0
        else:
           o2s_lesion = 1

        if 'p2s' in self.lesions:
           p2s_lesion = 0
        else:
           p2s_lesion = 1

        if 's2p' in self.lesions:
           s2p_lesion = 0
        else:
           s2p_lesion = 1

        if 's2s' in self.lesions:
           s2s_lesion = 0
        else:
           s2s_lesion = 1

        if 'p2p' in self.lesions:
           p2p_lesion = 0
        else:
           p2p_lesion = 1

        ### Compute lesioning adjustment
        a_p = max([1,p2p_lesion + s2p_lesion + 2 * o2p_lesion])
        a_s = max([1,s2s_lesion + p2s_lesion + 2 * o2s_lesion])

        ### Compute gradient of phonology
        phon_gradient = 0
        if p2p_lesion:
           phon_gradient = p2p_lesion * self.cleanup['hidden_to_state']['phonology'](cleanup_phon,phonology,a_p)
        if s2p_lesion:
           phon_gradient = phon_gradient + s2p_lesion * self.phonology_semantics['hidden_to_state']['phonology'](
                                                                    sem_2_phon,phonology,a_p)
        if o2p_lesion:
           phon_gradient = phon_gradient + o2p_lesion * (self.orthography_direct['phonology'](orthography,phonology,a_p) \
                               + self.orthography_indirect['hidden_to_state']['phonology'](orth_2_phon,phonology,a_p))

        ### Compute gradient of semantics
        sem_gradient = 0
        if s2s_lesion:
           sem_gradient = s2s_lesion * self.cleanup['hidden_to_state']['semantics'](cleanup_sem,semantics,a_s)
        if p2s_lesion:
           sem_gradient = sem_gradient + p2s_lesion * self.phonology_semantics['hidden_to_state']['semantics'](
                                                                    phon_2_sem,semantics,a_s)
        if o2s_lesion:
           sem_gradient = sem_gradient + o2s_lesion * (self.orthography_direct['semantics'](orthography,semantics,a_s) \
                               + self.orthography_indirect['hidden_to_state']['semantics'](orth_2_sem,semantics,a_s))

        ### Compute gradient of cleanup units
        cleanup_phon_gradient,cleanup_sem_gradient = 0,0
        
        if p2p_lesion:
           cleanup_phon_gradient = self.cleanup['state_to_hidden']['phonology'](phonology,cleanup_phon,1)
        if s2s_lesion:
           cleanup_sem_gradient = self.cleanup['state_to_hidden']['semantics'](semantics,cleanup_sem,1)

        ### Compute gradient of oral hidden units
        phon_2_sem_gradient,sem_2_phon_gradient = 0,0
        
        if p2s_lesion:
           phon_2_sem_gradient = self.phonology_semantics['state_to_hidden']['phonology'](phonology,phon_2_sem,1)
        if s2p_lesion:
           sem_2_phon_gradient = self.phonology_semantics['state_to_hidden']['semantics'](semantics,sem_2_phon,1)

        ### Compute gradient of reading hidden units
        orth_2_phon_gradient,orth_2_sem_gradient = 0,0

        if o2p_lesion:
           orth_2_phon_gradient = self.orthography_indirect['state_to_hidden']['phonology'](orthography,orth_2_phon,1)
        if o2s_lesion:
           orth_2_sem_gradient = self.orthography_indirect['state_to_hidden']['semantics'](orthography,orth_2_sem,1)

        ### Write gradients to dictionary
        gradients = {}

        gradients['orthography'] = torch.zeros_like(orthography,device=orthography.device)
        gradients['phonology'] = phon_gradient
        gradients['semantics'] = sem_gradient

        gradients['cleanup_phon'] = cleanup_phon_gradient
        gradients['cleanup_sem'] = cleanup_sem_gradient

        gradients['phon_2_sem'] = phon_2_sem_gradient
        gradients['sem_2_phon'] = sem_2_phon_gradient

        gradients['orth_2_sem'] = orth_2_sem_gradient
        gradients['orth_2_phon'] = orth_2_phon_gradient

        return gradients
