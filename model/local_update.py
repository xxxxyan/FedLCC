import random, torch
import torch.nn as nn
import torch.optim as optim

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr

def log_string(txt,out_str):
    txt.write(out_str+'\n')
    txt.flush()
    print(out_str)

class LocalUpdate(object):
    def __init__(self, args, user_train_data=None, dev_data=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.user_train_data = user_train_data 
        self.dev_data = dev_data
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        
    def train(self, init_lr, previous_model, global_model, model):
        parameters = filter(lambda p: p.requires_grad, model.parameters())

        if self.args.optimizer == "SGD":
            optimizer = optim.SGD(parameters, lr=init_lr, momentum=self.args.momentum, weight_decay=self.args.l2_penalty)
        elif self.args.optimizer == "Adam":
            optimizer = optim.Adam(parameters, lr=init_lr, weight_decay=self.args.l2_penalty)
        
        epoch_loss = []
        model.train()
        for local_epoch in range(self.args.local_epoch):
            batch_loss = []
            optimizer, lr = lr_decay(optimizer, local_epoch, self.args.lr_decay, init_lr)
            model.zero_grad()

            for step, batch in enumerate(self.user_train_data):
                inputs = {'input_ids': batch[0].to(self.args.device),
                        'attention_mask': batch[1].to(self.args.device),
                        'labels': batch[2],
                        'entity_pos': batch[3],
                        'hts': batch[4],
                        }
                outputs = model(**inputs)            
                logits = outputs[1]               
                
                labels = [torch.tensor(label) for label in batch[2]]
                labels = torch.cat(labels, dim=0).to(logits).long()
                labels = torch.argmax(labels, dim=-1)
                loss = self.loss_func(logits, labels)
                
                if self.args.fed_algo == 'MOON':
                    pro1 = outputs[0]
                    pro2 = global_model(**inputs)[0]
                    pro3 = previous_model(**inputs)[0]

                    posi = self.cos(pro1, pro2)
                    logits = posi.reshape(-1,1)
                
                    nega = self.cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=-1)
                    loss += self.args.moon_mu * self.loss_func(logits, torch.zeros(outputs[0].size(0)).cuda().long().to(logits.device))      
 
                loss.backward()
                if self.args.max_grad_norm>0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                if (step + 1) % self.args.local_gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()                    
            
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
                
        param = model.state_dict()

        for k, v in param.items():
            param[k] = v.cpu()
        return {"param": param, "loss": sum(epoch_loss) / len(epoch_loss), "final_lr": lr}