import copy
import torch, random
import torch.nn.functional as F

def server_opt(w_locals):
    w_avg = copy.deepcopy(w_locals[0])
    
    with torch.no_grad():
        for k in w_avg.keys():
            for i in range(1, len(w_locals)):
                w_avg[k] += w_locals[i][k]
            w_avg[k] = torch.true_divide(w_avg[k], len(w_locals))
            
    return w_avg

def aggregate_att(w_clients, w_server, stepsize,  metric=2, dp=0):
    
    para_position_ids = w_server['model.embeddings.position_ids']   
    w_server.popitem(last=False)
    for client in w_clients:
        client.popitem(last=False)
    
    w_next = copy.deepcopy(w_server)
    
    att, att_mat = {}, {}
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.norm(w_server[k]-w_clients[i][k], p=metric)
    for k in w_next.keys():
        att[k] = F.softmax(att[k], dim=0)
    for k in w_next.keys():
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
            w_next[k] = w_server[k] - torch.mul(att_weight, stepsize) + torch.mul(torch.randn(w_server[k].shape), dp)
    w_next['model.embeddings.position_ids'] = para_position_ids
    w_next.move_to_end('model.embeddings.position_ids', last=False)
    return w_next
