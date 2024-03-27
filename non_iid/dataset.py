import random
random.seed(0)
import numpy as np
np.random.seed(0)
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset


class DocRelationDataset:
    
    def __init__(self, loader, data_type, params, mappings):
        self.unk_w_prob = params['unk_w_prob']
        self.mappings = mappings
        self.loader = loader
        self.data_type = data_type
        self.edges = params['edges']
        self.data = []

    def __len__(self):
        return len(self.data)

    def __call__(self):
        pbar = tqdm(self.loader.documents.keys())
        for pmid in pbar:
            pbar.set_description('  Preparing {} data - PMID {}'.format(self.data_type.upper(), pmid))
            doc = []
            for sentence in self.loader.documents[pmid]:
                sent = []
                if self.data_type == 'train':
                    for w, word in enumerate(sentence):
                        if (word in self.mappings.singletons) and (random.uniform(0, 1) < float(self.unk_w_prob)):
                            sent += [self.mappings.word2index['<UNK>']]  # UNK words = singletons for train
                        else:
                            sent += [self.mappings.word2index[word]]

                else:

                    for w, word in enumerate(sentence):
                        if word in self.mappings.word2index:
                            sent += [self.mappings.word2index[word]]
                        else:
                            sent += [self.mappings.word2index['<UNK>']]
                assert len(sentence) == len(sent), '{}, {}'.format(len(sentence), len(sent))
                doc += [sent]
            
            nodes = []
            ent = []
            for id_, (e, i) in enumerate(self.loader.entities[pmid].items()):
                nodes += [[id_, self.mappings.type2index[i.type], int(i.mstart.split(':')[0]),
                           int(i.mend.split(':')[0]), i.sentNo.split(':')[0], 0]]
                
            for id_, (e, i) in enumerate(self.loader.entities[pmid].items()):
                for sent_id, m1, m2 in zip(i.sentNo.split(':'), i.mstart.split(':'), i.mend.split(':')):
                    ent += [[id_, self.mappings.type2index[i.type], int(m1), int(m2), int(sent_id)]]
                    nodes += [[id_, self.mappings.type2index[i.type], int(m1), int(m2), int(sent_id), 1]]

            for s, sentence in enumerate(self.loader.documents[pmid]):
                nodes += [[s, s, s, s, s, 2]]

            nodes = np.array(nodes, 'i')
            ent = np.array(ent, 'i')


            ents_keys = list(self.loader.entities[pmid].keys())  
            trel = -1 * np.ones((len(ents_keys), len(ents_keys)), 'i')
            rel_info = np.empty((len(ents_keys), len(ents_keys)), dtype='object_')
            for id_, (r, i) in enumerate(self.loader.pairs[pmid].items()):
                if i.type == 'not_include':
                    continue
                trel[ents_keys.index(r[0]), ents_keys.index(r[1])] = self.mappings.rel2index[i.type]
                rel_info[ents_keys.index(r[0]), ents_keys.index(r[1])] = OrderedDict(
                                                                         [('pmid', pmid),
                                                                          ('sentA', self.loader.entities[pmid][r[0]].sentNo),
                                                                          ('sentB',
                                                                           self.loader.entities[pmid][r[1]].sentNo),
                                                                          ('doc', self.loader.documents[pmid]),
                                                                          ('entA', self.loader.entities[pmid][r[0]]),
                                                                          ('entB', self.loader.entities[pmid][r[1]]),
                                                                          ('rel', self.mappings.rel2index[i.type]),
                                                                          ('dir', i.direction),
                                                                          ('cross', i.cross)])


            xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')

            r_id, c_id = nodes[xv, 5], nodes[yv, 5]
            r_Eid, c_Eid = nodes[xv, 0], nodes[yv, 0]
            r_Sid, c_Sid = nodes[xv, 4], nodes[yv, 4]
            
            adjacency = np.full((r_id.shape[0], r_id.shape[0]), 0, 'i')

            if 'FULL' in self.edges:
                adjacency = np.full(adjacency.shape, 1, 'i')

            if 'MM' in self.edges:
                # mention-mention
                adjacency = np.where((r_id == 1) & (c_id == 1) & (r_Sid == c_Sid), 1, adjacency)  # in same sentence

            if ('EM' in self.edges) or ('ME' in self.edges):
                # entity-mention
                adjacency = np.where((r_id == 0) & (c_id == 1) & (r_Eid == c_Eid), 1, adjacency)  # belongs to entity
                adjacency = np.where((r_id == 1) & (c_id == 0) & (r_Eid == c_Eid), 1, adjacency)

            if 'SS' in self.edges:
                # sentence-sentence (in order)
                adjacency = np.where((r_id == 2) & (c_id == 2) & (r_Sid == c_Sid - 1), 1, adjacency)
                adjacency = np.where((r_id == 2) & (c_id == 2) & (c_Sid == r_Sid - 1), 1, adjacency)

            if 'SS-ind' in self.edges:
                # sentence-sentence (direct + indirect)
                adjacency = np.where((r_id == 2) & (c_id == 2), 1, adjacency)

            if ('MS' in self.edges) or ('SM' in self.edges):
                # mention-sentence
                adjacency = np.where((r_id == 1) & (c_id == 2) & (r_Sid == c_Sid), 1, adjacency)  # belongs to sentence
                adjacency = np.where((r_id == 2) & (c_id == 1) & (r_Sid == c_Sid), 1, adjacency)

            if ('ES' in self.edges) or ('SE' in self.edges):
                # entity-sentence
                for x, y in zip(xv.ravel(), yv.ravel()):
                    if nodes[x, 5] == 0 and nodes[y, 5] == 2:  # this is an entity-sentence edge
                        z = np.where((r_Eid == nodes[x, 0]) & (r_id == 1) & (c_id == 2) & (c_Sid == nodes[y, 4]))

                        # at least one M in S
                        temp_ = np.where((r_id == 1) & (c_id == 2) & (r_Sid == c_Sid), 1, adjacency)
                        temp_ = np.where((r_id == 2) & (c_id == 1) & (r_Sid == c_Sid), 1, temp_)
                        adjacency[x, y] = 1 if (temp_[z] == 1).any() else 0
                        adjacency[y, x] = 1 if (temp_[z] == 1).any() else 0

            if 'EE' in self.edges:
                for i in range(len(trel)):
                    for j in range(len(trel)):
                        if trel[i, j] != -1:
                            adjacency[i, j] = 1
        

           
            adjacency[np.arange(r_id.shape[0]), np.arange(r_id.shape[0])] = 0

            if (trel == -1).all():  
                continue

            self.data += [{'pmid':pmid,
                           'nodes': nodes, 
                           'trel':trel,
                           'adjacency': adjacency}]
        return self.data
