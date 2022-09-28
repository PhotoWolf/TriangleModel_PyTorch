import torch
import pandas as pd
import numpy as np

class SlotTokenizer:
    def __init__(self,data_list,embedding_table = None):
        num_slots = len(data_list[0])
        slots = {slot:{} for slot in range(num_slots)}
        for word in data_list:
            for idx,char in enumerate(word):
                if char is '_':
                    continue;
                if slots[idx].get(char,False) is False:
                    slots[idx][char] = len(slots[idx])

        self.embedding_table = embedding_table
        self.slots = {slot:slots[slot] for slot in slots if len(slots[slot])}

        if self.embedding_table is not None:
            self.embedding_size = len(slots) * len(embedding_table.columns)
        else:
            self.embedding_size = sum([len(slots[slot]) for slot in slots])

    def __call__(self,word:str):
        embedding = torch.zeros((self.embedding_size))
        marker = 0
        for idx,char in enumerate(word):
            if idx not in self.slots:
                continue;

            if self.embedding_table is not None:
                embedding[marker:marker+len(self.embedding_table.columns)] = torch.FloatTensor(
                                                                        self.embedding_table.loc[char].to_numpy()
                                                                        )
                marker += len(self.embedding_table.columns)
            else:
                if char is not '_':
                    embedding[marker + self.slots[idx][char]] = 1 
                marker += len(self.slots[idx])
        return embedding


class Monosyllabic_Dataset(torch.utils.data.Dataset):
    def __init__(self,path_to_words,path_to_phon_mapping,path_to_sem):
        super(Monosyllabic_Dataset,self).__init__()

        data = pd.read_csv(path_to_words).drop_duplicates()
        self.orthography = data['ort']
        self.orthography_tokenizer = SlotTokenizer(self.orthography)

        self.phonology = data['pho']
        phon_mapping = pd.read_csv(path_to_phon_mapping,sep="\t",header=None).set_index(0)
        self.phonology_tokenizer = SlotTokenizer(self.phonology,phon_mapping)

        semantics = torch.FloatTensor(np.load(path_to_sem)['data'])
        self.semantics = semantics[:,(semantics==0).any(dim=0)]

        self.frequencies = np.clip(np.sqrt(data['wf'])/(30000**.5),.05,1)
        self.frequencies = self.frequencies/np.sum(self.frequencies)

    def __len__(self):
        return len(self.orthography)

    def __getitem__(self,idx):
        idx = np.random.choice(np.arange(self.__len__()),p=self.frequencies)
        
        orthography = self.orthography_tokenizer(self.orthography.iloc[idx])
        phonology = self.phonology_tokenizer(self.phonology.iloc[idx])
        semantics = self.semantics[idx]

        return {'orthography':orthography,'phonology':phonology,'semantics':semantics}
