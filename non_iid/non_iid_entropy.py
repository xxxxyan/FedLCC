from dataset import DocRelationDataset
from loader import DataLoader
import numpy as np
import json

parameters = dict()
parameters['train'] = './dataset/cdr/train.data'
parameters['label2ignore'] = '1:NR:2'
parameters['min_w_freq'] = 1
parameters['window'] = None
parameters['unk_w_prob'] = 0.5
parameters['edges'] = ['MM', 'ME', 'MS', 'ES', 'SS','EE'] 
     
train_loader = DataLoader(parameters['train'], parameters)
train_loader(embeds=None)
train_data = DocRelationDataset(train_loader, 'train', parameters, train_loader).__call__()

def calculate_structure_entropy(adj_matrix):
    node_degrees = np.sum(adj_matrix, axis=1)
    total_nodes = len(node_degrees)
    degree_probabilities = node_degrees / total_nodes
    entropy = -np.sum(degree_probabilities * np.log2(degree_probabilities))
    return entropy

for i, data in  enumerate(train_data):
    adj_matrix = data['adjacency']
    structure_entropy = calculate_structure_entropy(adj_matrix)
    train_data[i]['structure_entropy'] = structure_entropy 

######################################################################
# sorted_train_data = sorted(train_data, key=lambda  x: x['adjacency'].shape[0])
# min = min(train_data, key=lambda x: x['structure_entropy'])
# max = max(train_data, key=lambda x: x['structure_entropy'])

#CDR:4.317657676193542——24.289256154892264
#CHR:2.0262599019087846——17.025208239613598
#GDA:2.1128517833640057——30.8545658610087
######################################################################
partition = {}
for client_id in range(10):
    partition[client_id] = []
# ######################################################################
# #CDR
# for i, data in  enumerate(train_data):
#     s_e = data['structure_entropy']
#     if s_e <= 6:
#         partition[0].append(i)
#     elif 6< s_e <= 8:
#         partition[1].append(i)
#     elif 8< s_e <= 10:
#         partition[2].append(i)
#     elif 10< s_e <= 12:
#         partition[3].append(i)
#     elif 12< s_e <= 14:
#         partition[4].append(i)
#     elif 14< s_e <= 16:
#         partition[5].append(i)
#     elif 16< s_e <= 18:
#         partition[6].append(i)
#     elif 18< s_e <= 20:
#         partition[7].append(i)
#     elif 20< s_e <= 22:
#         partition[8].append(i)
#     else:
#         partition[9].append(i)
# ######################################################################
# #CHR
# for i, data in  enumerate(train_data):
#     s_e = data['structure_entropy']
#     if s_e <= 3.5:
#         partition[0].append(i)
#     elif 3.5< s_e <= 5.0:
#         partition[1].append(i)
#     elif 5.0< s_e <= 6.5:
#         partition[2].append(i)
#     elif 6.5< s_e <= 8.0:
#         partition[3].append(i)
#     elif 8.0< s_e <= 9.5:
#         partition[4].append(i)
#     elif 9.5< s_e <= 11.0:
#         partition[5].append(i)
#     elif 11.0< s_e <= 12.5:
#         partition[6].append(i)
#     elif 12.5< s_e <= 14.0:
#         partition[7].append(i)
#     elif 14.0< s_e <= 15.5:
#         partition[8].append(i)
#     else:
#         partition[9].append(i)
######################################################################
# #GDA
# for i, data in  enumerate(train_data):
#     s_e = data['structure_entropy']
#     if s_e <= 4.8:
#         partition[0].append(i)
#     elif 4.8< s_e <= 7.6:
#         partition[1].append(i)
#     elif 7.6< s_e <= 10.4:
#         partition[2].append(i)
#     elif 10.4< s_e <= 13.2:
#         partition[3].append(i)
#     elif 13.2< s_e <= 16.0:
#         partition[4].append(i)
#     elif 16.0< s_e <= 18.8:
#         partition[5].append(i)
#     elif 18.8< s_e <= 21.6:
#         partition[6].append(i)
#     elif 21.6< s_e <= 24.4:
#         partition[7].append(i)
#     elif 24.4< s_e <= 27.2:
#         partition[8].append(i)
#     else:
#         partition[9].append(i)

# with open('./dataset/cdr/cdr_non_iid.json', "w") as json_file:
#     json.dump(partition, json_file)