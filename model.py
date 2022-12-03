import torch
import json
from typing import Optional,List,Dict
from dataclasses import dataclass

class TimeAveragedInputs(torch.nn.Module):
   def __init__(self,in_features_list : List[int], out_features : int,bias : Optional[bool] =False):
       '''
        Time-Averaged Inputs formulation of the gradient (refer to
        Plaut et. al, 1998).

        Args:
           in_features_list (List[int]) : dimemsionality of each input vector
           out_features (int) : dimensionality of output vectors
           bias (bool) : if True, use trainable bias term

       '''
       super(TimeAveragedInputs,self).__init__()
       self.weights = torch.nn.ModuleList(
                          [torch.nn.Linear(in_features,out_features,bias=bias) for in_features in in_features_list]   
                          )

   def forward(self, X : List[torch.Tensor], Y : torch.Tensor) -> torch.Tensor:
       '''
         Compute the gradient of [Y] w.r.t to time. 
         
         Args:
            X (List[torch.Tensor]) : dimensionalities [in_features_list]
            Y (torch.Tensor) : dimensionality [out_features]
         Returns:
            gradient contribution; dimensionality [out_features]
       '''
       nX = 0
       for idx,input_vector in enumerate(X):
           if torch.isinf(input_vector).all(): continue;
           nX = nX + self.weights[idx](torch.sigmoid(input_vector))
       return nX - Y


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

        ### Instantiate phonology gradient
        self.phon_gradient = operator([phon_cleanup_dim,sem_2_phon_dim,
                                    orth_2_phon_dim,orth_dim],phon_dim,learn_bias)

        ### Instantiate semantics gradient
        self.sem_gradient = operator([sem_cleanup_dim,phon_2_sem_dim,
                                    orth_2_sem_dim,orth_dim],sem_dim,learn_bias)

        ### Instantiate cleanup gradients
        self.p2p_gradient = operator([phon_dim],phon_cleanup_dim,learn_bias)
        self.s2s_gradient = operator([sem_dim],sem_cleanup_dim,learn_bias)

        ### Instantiate oral hidden unit gradients
        self.s2p_gradient = operator([sem_dim],sem_2_phon_dim,learn_bias)
        self.p2s_gradient = operator([phon_dim],phon_2_sem_dim,learn_bias)

        ### Instantiate reading hidden unit gradients
        self.o2p_gradient = operator([orth_dim],orth_2_phon_dim,learn_bias)
        self.o2s_gradient = operator([orth_dim],orth_2_sem_dim,learn_bias)


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

        suppress_inputs = lambda inputs,lam: -float("Inf") * torch.ones_like(inputs,device=inputs.device) if lam == 0 else inputs

        ### Compute gradient of phonology
        phon_gradient = 0
        if (p2p_lesion + s2p_lesion + o2p_lesion) > 0:
           phon_gradient = self.phon_gradient([suppress_inputs(cleanup_phon,p2p_lesion),
                                                  suppress_inputs(sem_2_phon,s2p_lesion),
                                                  suppress_inputs(orth_2_phon,o2p_lesion),
                                                  suppress_inputs(orthography,o2p_lesion)],
                                               phonology)
        ### Compute gradient of semantics
        sem_gradient = 0
        if (s2s_lesion + p2s_lesion + o2s_lesion) > 0:
           sem_gradient = self.sem_gradient([suppress_inputs(cleanup_sem,s2s_lesion),
                                                suppress_inputs(phon_2_sem,p2s_lesion),
                                                suppress_inputs(orth_2_sem,o2s_lesion),
                                                suppress_inputs(orthography,o2s_lesion)],
                                              semantics)

        ### Compute gradient of cleanup units
        cleanup_phon_gradient,cleanup_sem_gradient = 0,0
        if p2p_lesion:
           cleanup_phon_gradient = self.p2p_gradient([suppress_inputs(phonology,p2p_lesion)],cleanup_phon)
        if s2s_lesion:
           cleanup_sem_gradient = self.s2s_gradient([suppress_inputs(semantics,s2s_lesion)],cleanup_sem)

        ### Compute gradient of oral hidden units
        phon_2_sem_gradient,sem_2_phon_gradient = 0,0
        if p2s_lesion:
           phon_2_sem_gradient = self.p2s_gradient([suppress_inputs(phonology,p2s_lesion)],phon_2_sem)
        if s2p_lesion:
           sem_2_phon_gradient = self.s2p_gradient([suppress_inputs(semantics,s2p_lesion)],sem_2_phon)

        ### Compute gradient of reading hidden units
        orth_2_sem_gradient,orth_2_phon_gradient = 0,0
        if o2s_lesion:
           orth_2_sem_gradient = self.o2s_gradient([suppress_inputs(orthography,o2s_lesion)],orth_2_sem)
        if o2p_lesion:
           orth_2_phon_gradient = self.o2p_gradient([suppress_inputs(orthography,o2p_lesion)],orth_2_phon)

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
