import torch
from dataclasses import dataclass

@dataclass()
class ModelConfig:

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

    def create_model(self,lesions=[]):
        return TriangleModel(self.orth_dim,self.phon_dim,self.sem_dim,
                                self.phon_cleanup_dim,self.sem_cleanup_dim,
                                self.phon_2_sem_dim,self.sem_2_phon_dim,
                                self.orth_2_sem_dim,self.orth_2_phon_dim,
                                self.learn_bias,lesions)

class TimeAveragedInputs(torch.nn.Module):
   def __init__(self,in_features,out_features,bias=False):
       super(TimeAveragedInputs,self).__init__()
       self.W1 = torch.nn.Linear(in_features,out_features,bias=bias)

   def forward(self,X,Y,s):
       return self.W1(torch.sigmoid(X)) - (1/s) * Y

class TriangleModel(torch.nn.Module):
    def __init__(self,orth_dim,phon_dim,sem_dim,
                    phon_cleanup_dim,sem_cleanup_dim,
                    phon_2_sem_dim,sem_2_phon_dim,
                    orth_2_sem_dim,orth_2_phon_dim,
                    learn_bias,lesions):
        super(TriangleModel,self).__init__()
        '''
          A PyTorch implemtation of the Triangle Model discussed in
          Harm and Seidenberg, 2004.
        '''
        self.lesions = lesions

        self.orth_dim,self.phon_dim,self.sem_dim = orth_dim,phon_dim,sem_dim
        self.phon_cleanup_dim,self.sem_cleanup_dim = phon_cleanup_dim,sem_cleanup_dim
        self.phon_2_sem_dim,self.sem_2_phon_dim = phon_2_sem_dim,sem_2_phon_dim
        self.orth_2_sem_dim,self.orth_2_phon_dim = orth_2_sem_dim,orth_2_phon_dim

        self.cleanup = torch.nn.ModuleDict({'state_to_hidden':torch.nn.ModuleDict({
                                        'semantics':TimeAveragedInputs(sem_dim,sem_cleanup_dim,learn_bias),
                                        'phonology':TimeAveragedInputs(phon_dim,phon_cleanup_dim,learn_bias)}),
                                            'hidden_to_state':torch.nn.ModuleDict({
                                        'semantics':TimeAveragedInputs(sem_cleanup_dim,sem_dim,learn_bias),
                                        'phonology':TimeAveragedInputs(phon_cleanup_dim,phon_dim,learn_bias)})
                                        })
        self.phonology_semantics = torch.nn.ModuleDict({'state_to_hidden':torch.nn.ModuleDict({
                                                 'semantics':TimeAveragedInputs(sem_dim,sem_2_phon_dim,learn_bias),
                                                 'phonology':TimeAveragedInputs(phon_dim,phon_2_sem_dim,learn_bias)}),
                                                        'hidden_to_state':torch.nn.ModuleDict({
                                                 'phonology':TimeAveragedInputs(sem_2_phon_dim,phon_dim,learn_bias),
                                                 'semantics':TimeAveragedInputs(phon_2_sem_dim,sem_dim,learn_bias)})
                                                 })
        self.orthography_indirect = torch.nn.ModuleDict({'state_to_hidden':torch.nn.ModuleDict({
                                                  'semantics':TimeAveragedInputs(orth_dim,orth_2_sem_dim,learn_bias), 
                                                  'phonology':TimeAveragedInputs(orth_dim,orth_2_phon_dim,learn_bias)}),
                                                        'hidden_to_state':torch.nn.ModuleDict({
                                                  'semantics':TimeAveragedInputs(orth_2_sem_dim,sem_dim,learn_bias),
                                                  'phonology':TimeAveragedInputs(orth_2_phon_dim,phon_dim,learn_bias)})
                                                  })
        self.orthography_direct = torch.nn.ModuleDict({
                                                   'semantics':TimeAveragedInputs(orth_dim,sem_dim,learn_bias),
                                                   'phonology':TimeAveragedInputs(orth_dim,phon_dim,learn_bias)}
                                                   )
    def forward(self,inputs):

        orthography = inputs['orthography']
        phonology = inputs['phonology']
        semantics = inputs['semantics'] 

        cleanup_phon = inputs['cleanup_phon']
        cleanup_sem = inputs['cleanup_sem']

        phon_2_sem = inputs['phon_2_sem']
        sem_2_phon = inputs['sem_2_phon']

        orth_2_sem = inputs['orth_2_sem']
        orth_2_phon = inputs['orth_2_phon']

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

        a_p = max([1,p2p_lesion + s2p_lesion + 2 * o2p_lesion])
        a_s = max([1,s2s_lesion + p2s_lesion + 2 * o2s_lesion])

        ### Compute gradient of phonology state w.r.t time
        phon_gradient = p2p_lesion * self.cleanup['hidden_to_state']['phonology'](cleanup_phon,phonology,a_p)
        phon_gradient = phon_gradient + s2p_lesion * self.phonology_semantics['hidden_to_state']['phonology'](
                                                                    sem_2_phon,phonology,a_p
                                                               )
        phon_gradient = phon_gradient + o2p_lesion * (self.orthography_direct['phonology'](orthography,phonology,a_p) \
                               + self.orthography_indirect['hidden_to_state']['phonology'](orth_2_phon,phonology,a_p))

#        print(phon_gradient.abs().max())
        ### Compute gradient of semantic state w.r.t time
        sem_gradient = s2s_lesion * self.cleanup['hidden_to_state']['semantics'](cleanup_sem,semantics,a_s)
        sem_gradient = sem_gradient + p2s_lesion * self.phonology_semantics['hidden_to_state']['semantics'](
                                                                    phon_2_sem,semantics,a_s
                                                               )
        sem_gradient = sem_gradient + o2s_lesion * (self.orthography_direct['semantics'](orthography,semantics,a_s) \
                               + self.orthography_indirect['hidden_to_state']['semantics'](orth_2_sem,semantics,a_s))

        ### Compute gradient of cleanup units w.r.t time
        cleanup_phon_gradient = self.cleanup['state_to_hidden']['phonology'](phonology,cleanup_phon,1)
        cleanup_sem_gradient = self.cleanup['state_to_hidden']['semantics'](semantics,cleanup_sem,1)

        ### Compute gradient of p->s and s->p units w.r.t time
        phon_2_sem_gradient = self.phonology_semantics['state_to_hidden']['phonology'](phonology,phon_2_sem,1)
        sem_2_phon_gradient = self.phonology_semantics['state_to_hidden']['semantics'](semantics,sem_2_phon,1)

        ### Compute gradient of orth indirect units w.r.t time
        orth_2_phon_gradient = self.orthography_indirect['state_to_hidden']['phonology'](orthography,orth_2_phon,1)
        orth_2_sem_gradient = self.orthography_indirect['state_to_hidden']['semantics'](orthography,orth_2_sem,1)

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
