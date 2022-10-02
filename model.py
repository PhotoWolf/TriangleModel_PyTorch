import torch
from dataclasses import dataclass

from dataclasses import dataclass

@dataclass()
class ModelConfig:

    init_states : bool = False

    orth_dim : int = 111
    phon_dim : int = 200
    sem_dim : int = 1989

    phon_cleanup_dim : int = 50
    sem_cleanup_dim : int = 50

    phon_2_sem_dim : int = 500
    sem_2_phon_dim : int = 500

    orth_2_sem_dim : int = 500
    orth_2_phon_dim : int = 100
	
    def create_model(self):
        return TriangleModel(self.orth_dim,self.phon_dim,self.sem_dim,
                                self.phon_cleanup_dim,self.sem_cleanup_dim,
                                self.phon_2_sem_dim,self.sem_2_phon_dim,
                                self.orth_2_sem_dim,self.orth_2_phon_dim,
                                self.init_states)

def st_clamp(x,bound=15):
    clamp_x = torch.clamp(x,-bound,bound)
    return clamp_x.detach() + x - x.detach()

def scaled_tanh(x,bound=15):
    return bound * torch.tanh(x)

def sigmoid_inverse(y,t = 1,bound=15):
    y_prime = -(1/y-1).log() / t
    y_prime[y>0.999999] = bound
    y_prime[y<0.000001] = -bound
    return y_prime

class TimeAveragedInputs(torch.nn.Module):
    def __init__(self,in_features,out_features,bias=False):
      super(TimeAveragedInputs,self).__init__()
      self.f = torch.nn.Linear(in_features,out_features)

    def forward(self,X,Y):
        X_prime = st_clamp(self.f(X))
        return X_prime-Y

class TriangleModel(torch.nn.Module):
    def __init__(self,orth_dim,phon_dim,sem_dim,
                    phon_cleanup_dim,sem_cleanup_dim,
                    phon_2_sem_dim,sem_2_phon_dim,
                    orth_2_sem_dim,orth_2_phon_dim,
                    init_states):
        super(TriangleModel,self).__init__()
        '''
          A PyTorch implemtation of the Triangle Model discussed in
          Harm and Seidenberg, 2004.
        '''
        self.orth_dim,self.phon_dim,self.sem_dim = orth_dim,phon_dim,sem_dim
        self.phon_cleanup_dim,self.sem_cleanup_dim = phon_cleanup_dim,sem_cleanup_dim
        self.phon_2_sem_dim,self.sem_2_phon_dim = phon_2_sem_dim,sem_2_phon_dim
        self.orth_2_sem_dim,self.orth_2_phon_dim = orth_2_sem_dim,orth_2_phon_dim
	
	self.cleanup = torch.nn.ModuleDict({'state_to_hidden':torch.nn.ModuleDict({
                                                                 'semantics':TimeAveragedInputs(sem_dim,sem_cleanup_dim),
                                                                 'phonology':TimeAveragedInputs(phon_dim,phon_cleanup_dim)}
                                                                ),
                                            'hidden_to_state':torch.nn.ModuleDict({
                                                                 'semantics':TimeAveragedInputs(sem_cleanup_dim,sem_dim),
                                                                 'phonology':TimeAveragedInputs(phon_cleanup_dim,phon_dim)}
                                                                )}
                                           )
        self.phonology_semantics = torch.nn.ModuleDict({'state_to_hidden':torch.nn.ModuleDict({
                                                                             'semantics':TimeAveragedInputs(sem_dim,sem_2_phon_dim),
                                                                             'phonology':TimeAveragedInputs(phon_dim,phon_2_sem_dim)}
                                                                            ),
                                                        'hidden_to_state':torch.nn.ModuleDict({
                                                                             'phonology':TimeAveragedInputs(sem_2_phon_dim,phon_dim),
                                                                             'semantics':TimeAveragedInputs(phon_2_sem_dim,sem_dim)}
                                                                            )}
                                                       )
	self.orthography_indirect = torch.nn.ModuleDict({'state_to_hidden':torch.nn.ModuleDict({
                                                                              'semantics':TimeAveragedInputs(orth_dim,orth_2_sem_dim),
                                                                              'phonology':TimeAveragedInputs(orth_dim,orth_2_phon_dim)}
                                                                             ),
                                                        'hidden_to_state':torch.nn.ModuleDict({
                                                                             'semantics':TimeAveragedInputs(orth_2_sem_dim,sem_dim),
                                                                             'phonology':TimeAveragedInputs(orth_2_phon_dim,phon_dim)}
                                                                            )}
                                                        )
        self.orthography_direct = torch.nn.ModuleDict({
                                                       'semantics':TimeAveragedInputs(orth_dim,sem_dim),
                                                       'phonology':TimeAveragedInputs(orth_dim,phon_dim)}
                                                       )
	self.default_inputs = torch.nn.ParameterDict({
                                                        'semantics':torch.nn.Parameter(torch.zeros((1,sem_dim)),requires_grad=init_states),
                                                        'phonology':torch.nn.Parameter(torch.zeros((1,phon_dim)),requires_grad=init_states),
                                                        'orthography':torch.nn.Parameter(torch.zeros((1,orth_dim)),requires_grad=init_states)}
                                                     )
	 self.default_hidden = torch.nn.ModuleDict({
                           'cleanup':torch.nn.ParameterDict({
                              'semantics':torch.nn.Parameter(torch.zeros((1,sem_cleanup_dim)),requires_grad=init_states),
                              'phonology':torch.nn.Parameter(torch.zeros((1,phon_cleanup_dim)),requires_grad=init_states)
                           }),
                           'phonology_semantics':torch.nn.ParameterDict({
                              'semantics':torch.nn.Parameter(torch.zeros((1,sem_2_phon_dim)),requires_grad=init_states),
                              'phonology':torch.nn.Parameter(torch.zeros((1,phon_2_sem_dim)),requires_grad=init_states)
                           }),
                           'orthography':torch.nn.ParameterDict({
                              'semantics':torch.nn.Parameter(torch.zeros((1,orth_2_sem_dim)),requires_grad=init_states),
                              'phonology':torch.nn.Parameter(torch.zeros((1,orth_2_phon_dim)),requires_grad=init_states)
                           })
         })

    def forward(self,start_time,T,eta = 1e-1,orthography=None,phonology=None,semantics=None,lesions=[]):
        '''
          Depending on the combination of inputs, _forward_ will choose one of
          four possible systems.

              [orthography]: o->s/p, s<->p, s->s, and p->p. This correspponds to
                             the full system of coupled ODEs.
              [phonology]: p->s and s->s.
              [semantics]: s->p and p->p.
              [phonology],[semantics]: s->s and p->p. Only cleanup units.
        '''

        assert (orthography is not None or phonology is not None or semantics is not None)
	orthography_mask = int((orthography is not None))
        phonology_mask = int((phonology is not None))
        semantics_mask = int((semantics is not None))

        if 'o2p' in lesions:
           o2p_lesion = 0
        else:
           o2p_lesion = 1

        if 'o2s' in lesions:
           o2s_lesion = 0
        else:
           o2s_lesion = 1

        if 'p2s' in lesions:
           p2s_lesion = 0
        else:
           p2s_lesion = 1

        if 's2p' in lesions:
           s2p_lesion = 0
        else:
           s2p_lesion = 1
	
	if phonology_mask and semantics_mask:
            phonology_mask = 0
            semantics_mask = 0

        if orthography_mask:
            phonology_mask = 1
            semantics_mask = 1

        data_list = [semantics,phonology,orthography]
        for idx,data in enumerate(data_list):
            if data is not None:
                batch_size = data.shape[0]
                device = data.device
                break;

        data_types = ['semantics','phonology','orthography']
        for idx in range(len(data_list)):
            if data_list[idx] is None:
                data_list[idx] = 1 * self.default_inputs[data_types[idx]].repeat(batch_size,1).to(device)
		
	time = 0
        hidden_units = [
                          [sigmoid_inverse(1 * self.default_hidden['cleanup']['semantics']),
                           sigmoid_inverse(1 * self.default_hidden['cleanup']['phonology'])
                          ],
                          [sigmoid_inverse(1 * self.default_hidden['phonology_semantics']['semantics']),
                           sigmoid_inverse(1 * self.default_hidden['phonology_semantics']['phonology'])
                          ],
                          [sigmoid_inverse(1 * self.default_hidden['orthography']['semantics']),
                           sigmoid_inverse(1 * self.default_hidden['orthography']['phonology'])
                          ],
        ]

        hidden_list = [
                         [torch.sigmoid(hidden_units[0][0]),torch.sigmoid(hidden_units[0][1])],
                         [torch.sigmoid(hidden_units[1][0]),torch.sigmoid(hidden_units[1][1])],
                         [torch.sigmoid(hidden_units[2][0]),torch.sigmoid(hidden_units[2][1])],
        ]
	
	semantics,phonology,orthography = data_list
        semantics = sigmoid_inverse(semantics)
        phonology = sigmoid_inverse(phonology)

        output_list = [torch.sigmoid(phonology[None]),torch.sigmoid(semantics[None])]
        contributions = [[],[]]
	
	while True:

            nabla_s_clean = self.cleanup['hidden_to_state']['semantics'](hidden_list[0][0],semantics)
            nabla_s_phon = p2s_lesion * phonology_mask * \
                self.phonology_semantics['hidden_to_state']['semantics'](hidden_list[1][1],semantics)
            nabla_s_orth = o2s_lesion * orthography_mask * \
                (self.orthography_indirect['hidden_to_state']['semantics'](hidden_list[2][0],semantics)\
                           + self.orthography_direct['semantics'](orthography,semantics)
                        )

            nabla_s = nabla_s_clean + nabla_s_phon + nabla_s_orth
            contributions[1].append([nabla_s_clean.abs().mean().item(),
                                     nabla_s_phon.abs().mean().item(),
                                     nabla_s_orth.abs().mean().item()]
                                   )
	
	    nabla_p_clean = self.cleanup['hidden_to_state']['phonology'](hidden_list[0][1],phonology)
            nabla_p_sem = s2p_lesion * semantics_mask * \
                 self.phonology_semantics['hidden_to_state']['phonology'](hidden_list[1][0],phonology)
            nabla_p_orth = o2p_lesion * orthography_mask * \
                 (self.orthography_indirect['hidden_to_state']['phonology'](hidden_list[2][1],phonology)\
                                  + self.orthography_direct['phonology'](orthography,phonology)
                        )

            nabla_p = nabla_p_clean + nabla_p_sem + nabla_p_orth
            contributions[0].append([nabla_p_clean.abs().mean().item(),
                                     nabla_p_sem.abs().mean().item(),
                                     nabla_p_orth.abs().mean().item()]
                                   )
	    nabla_h = [
                          [
                              self.cleanup['state_to_hidden']['semantics'](output_list[1][-1],hidden_units[0][0]),
                              self.cleanup['state_to_hidden']['phonology'](output_list[0][-1],hidden_units[0][1])
                          ],
                          [
                              self.phonology_semantics['state_to_hidden']['semantics'](output_list[1][-1],hidden_units[1][0]),
                              self.phonology_semantics['state_to_hidden']['phonology'](output_list[0][-1],hidden_units[1][1])
                          ],
                          [
                              self.orthography_indirect['state_to_hidden']['semantics'](orthography,hidden_units[2][0]),
                              self.orthography_indirect['state_to_hidden']['phonology'](orthography,hidden_units[2][1])
                          ]
                       ]

            if time>=start_time:
                if orthography_mask == 1 or phonology_mask == 0:
                   phonology = phonology + eta * nabla_p
                if orthography_mask == 1 or semantics_mask == 0:
                   semantics = semantics + eta * nabla_s
		
	    output_list[0] = torch.cat((output_list[0],torch.sigmoid(phonology[None])),dim=0)
            output_list[1] = torch.cat((output_list[1],torch.sigmoid(semantics[None])),dim=0)

            for idx in range(len(hidden_units)):
                inner = []
                for jdx in range(len(hidden_units[idx])):
                    hidden_units[idx][jdx] =  hidden_units[idx][jdx] + eta * nabla_h[idx][jdx]
                    hidden_list[idx][jdx] = torch.sigmoid(hidden_units[idx][jdx])


            time += eta
            if time >= T:
               return output_list[0],output_list[1],contributions
	
