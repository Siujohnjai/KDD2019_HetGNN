from collections import Counter
from torch.utils import data
from typing import Dict, Tuple
import re 
import torch

Mapping = Dict[str, int]


def create_mappings(dataset_path: str) -> Tuple[Mapping, Mapping]:
    """Creates separate mappings to indices for entities and relations."""
    # counters to have entities/relations sorted from most frequent
    entity_counter = Counter()
    relation_counter = Counter()
    with open(dataset_path, "r") as f:
        for line in f:
            # -1 to remove newline sign
            head, relation, tail = line[:-1].split("\t")
            entity_counter.update([head, tail])
            relation_counter.update([relation])
    entity2id = {}
    relation2id = {}
    for idx, (mid, _) in enumerate(entity_counter.most_common()):
        entity2id[mid] = idx
    for idx, (relation, _) in enumerate(relation_counter.most_common()):
        relation2id[relation] = idx
    print(entity2id)
    return entity2id, relation2id


class FB15KDataset(data.Dataset):
    """Dataset implementation for handling FB15K and FB15K-237."""

    def __init__(self, data_path: str, relation_f):
        # self.entity2id = entity2id
        # self.relation2id = relation2id
        # with open(data_path, "r") as f:
            # data in tuples (head, relation, tail)
            # self.data = [line[:-1].split("\t") for line in f]
        self.data = []
        for i in range(len(relation_f)):
            f_name = relation_f[i]
            neigh_f = open(data_path + f_name, "r")
            sub_data_r = [] # data with same relation type r 
            for line in neigh_f:   
                line = line.strip()
                node_id = int(re.split(':', line)[0])
                neigh_list = re.split(':', line)[1]
                neigh_list_id = re.split(',', neigh_list)
                for neigh_id in neigh_list_id:
                    sub_data_r.append([node_id, i, int(neigh_id)])
            
            self.data.append(sub_data_r)
        
        print('length: ', len(self.data))
        # print(self.data[3])
        
    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data[0])

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        # x = torch.zeros([len(self.data), 3])
        x = []
        for i in range(len(self.data)): #5
            # print("data_index", self.data[i][index])
            try:
                x.append(self.data[i][index])
            except:
                x.append([99999,99999,99999])
        # head_id = self._to_idx(head, self.entity2id)
        # relation_id = self._to_idx(relation, self.relation2id)
        # tail_id = self._to_idx(tail, self.entity2id)
        return torch.LongTensor(x)

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)
