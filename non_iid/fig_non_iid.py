import json
import pickle
import matplotlib.pyplot as plt

dataset = 'cdr'
 
with open('/raid/xiaoyan/FedLCC/dataset/%s/%s_non_iid.json'% (dataset,dataset), "r") as json_file: 
    data = json.load(json_file)   

partition_nums = []
for client in data:
    partition_nums.append(len(data[client])) 
plt.hist(partition_nums)
plt.title('GDA')
plt.xlabel("number of samples")
plt.ylabel("number of clients")
plt.savefig('./dataset/%s/%s_non_iid.png'% (dataset,dataset), dpi=300)
print('ok')