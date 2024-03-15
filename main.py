import argparse, torch, pickle, os, time, sys, copy
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer

from model.model import Model
from model.local_update import LocalUpdate
from model.aggregation import server_opt, aggregate_att

from utils.prepro import read_bio
from utils.sampling import *

import datetime
from scipy import signal

def log_string(txt,out_str):
    txt.write(out_str+'\n')
    txt.flush()
    print(out_str)
    
def str2bool(v):
    return v.lower() in ('true')

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts)
    return output

def evaluate(args, model, features, tag="dev"):
    
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds, golds = [], []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            outputs = model(**inputs)                
            logits = outputs[1]
            
            predict = torch.argmax(logits, dim=-1)
            pred = torch.nn.functional.one_hot(predict, args.num_class).float()   
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            golds.append(np.concatenate([np.array(label, np.float32) for label in batch[2]], axis=0))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)

    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum()
    tn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).astype(np.float32).sum()
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + tn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    output = {
        "{}_precision".format(tag): precision * 100,
        "{}_recall".format(tag): recall * 100,
        "{}_f1".format(tag): f1 * 100,
    }
    return f1, output

def nn_train(args, log_txt, model, train_features, dev_features, test_features):
    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() 
                    if not any(nd in n for nd in new_layer)], 'weight_decay': args.l2_penalty},
        {"params": [p for n, p in model.named_parameters() 
                    if any(nd in n for nd in new_layer)], 'weight_decay': 0.0}
    ]

    if args.optimizer == "Adam":
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum)
                         
    model.zero_grad()
    best_dev = -1
    best_test = -1
    train_dataloader = DataLoader(train_features, batch_size=args.nn_train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
     
    for epoch in range(int(args.num_train_epochs)):
        log_string(log_txt,"Epoch: %s/%s" % (epoch, int(args.num_train_epochs)))
        sample_loss = 0
        total_loss = 0
        model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                        'attention_mask': batch[1].to(args.device),
                        'labels': batch[2],
                        'entity_pos': batch[3],
                        'hts': batch[4],
                        }
            outputs = model(**inputs)#(bl,) + (logits,)                 
            logits = outputs[1]
            labels = [torch.tensor(label) for label in batch[2]]
            labels = torch.cat(labels, dim=0).to(logits).long()
            labels = torch.argmax(labels, dim=-1)#将onehot转label
            loss = nn.CrossEntropyLoss()(logits, labels)
                
            sample_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
            end = (step + 1) * args.nn_train_batch_size
            if (end + 1) % 10000 == 0:
                sample_loss = 0
       
        log_string(log_txt,"Epoch: %s training finished. total loss: %s, then Test on Devdata!" % (epoch, total_loss))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        log_string(log_txt,str(dev_output))
        if dev_score > best_dev:
            log_string(log_txt,"Current best dev f1 score is %.4f, Exceed previous best dev f1 score %.4f" % (dev_score, best_dev))
            best_dev = dev_score
            test_score, test_output = evaluate(args, model, test_features, tag="test")
            log_string(log_txt,str(test_output))            
            if args.save_path != "":
                save_path = os.path.join(args.save_path + args.dataset_name, args.fed_algo + "_param")                                        
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_model="{}epoch_{}_test_f1_{}.model".format(save_path+'/',epoch, best_dev)             
                torch.save(model.state_dict(), save_model)

                
def fed_train(args, log_txt, model, train_features, dev_features, test_features):  

    init_lr = args.lr
    dict_users = iid_sampling(train_features, args.num_users) 
    net_glob = copy.deepcopy(model)
    net_glob.train()
    
    net_locals = [copy.deepcopy(net_glob) for i in range(args.num_users)]
    
    train_loss = []
    global_best_dev_f1 = -1
    global_best_test_f1 = -1
    for iter in range(int(args.num_train_epochs)):
        if iter != 0:
            init_lr = init_lr * ((1-args.lr_decay))
        log_string(log_txt, "Epoch: %s/%s; lr:%s" % (iter, args.num_train_epochs,  str(init_lr)))
        w_locals, local_loss = [], []

        m = max(int(args.frac * args.num_users), 1)
        user_idxs = np.random.choice(range(args.num_users), m, replace=False)
        for idxs in user_idxs:
            user_train_features = [train_features[idx] for idx in dict_users[idxs]]
            user_train_dataloader = DataLoader(user_train_features, batch_size=args.local_train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
            local_init = {"args": args, "user_train_data":user_train_dataloader, "dev_data": dev_features}                                    
            local = LocalUpdate(**local_init)
            
            if args.fed_algo == 'fedlcc':            
                local_out = local.train(init_lr, 
                                        previous_model = copy.deepcopy(net_locals[idxs]).to(args.device), 
                                        global_model = copy.deepcopy(net_glob).to(args.device), 
                                        model = copy.deepcopy(net_glob).to(args.device))                            
                net_locals[idxs].load_state_dict((local_out["param"]))    
                                    
            else:       
                local_model = copy.deepcopy(net_glob).to(args.device)        
                local_out = local.train(init_lr,
                                        previous_model = None,
                                        global_model = None,
                                        model = local_model)
                
            w_locals.append(copy.deepcopy(local_out["param"]))
            local_loss.append(copy.deepcopy(local_out["loss"]))
            init_lr = local_out["final_lr"]
            
        if args.fed_algo == "fed_attn":
            w_glob = net_glob.state_dict()
            for k, v in w_glob.items():
                w_glob[k] = v.cpu()
            w_glob = aggregate_att(w_locals, w_glob, args.stepsize, args.metric, dp=args.dp)
        else:           
            w_glob = server_opt(w_locals)

        net_glob.load_state_dict(w_glob)           
                
        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)  
        
        
        log_string(log_txt, "Epoch: %s training finished. Train loss: %s, then Test on Devdata!" % (iter, loss_avg)) 
        dev_score, dev_output = evaluate(args, net_glob, dev_features, tag="dev")
        log_string(log_txt,str(dev_output))
        if dev_score > global_best_dev_f1:
            log_string(log_txt,"Current best dev f1 score is %.4f, Exceed previous best test f1 score %.4f" % (dev_score, global_best_dev_f1))
            test_score, test_output = evaluate(args, net_glob, test_features, tag="test")
            log_string(log_txt,str(test_output))            
            if args.save_path != "":
                save_path = os.path.join(args.save_path + args.dataset_name, args.fed_algo + "_param")                                        
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_model="{}epoch_{}_dev_f1_{}.model".format(save_path+'/',iter, global_best_dev_f1)             
                torch.save(net_glob.state_dict(), save_model)
                global_best_dev_f1 = dev_score
        
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
    parser.add_argument('--dataset_name', type=str, choices=['chr', 'cdr', 'gda'], default='gda')
    parser.add_argument("--data_dir", default="./dataset/", type=str)
    parser.add_argument("--save_path", default="./save_model/", type=str)
    
    parser.add_argument("--train_file", default="train.data", type=str)
    parser.add_argument("--dev_file", default="dev.data", type=str)
    parser.add_argument("--test_file", default="test.data", type=str)
    
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default="./scibert_scivocab_cased", type=str)
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of relation types in dataset.")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    parser.add_argument("--fed_algo", type=str, choices=['None', 'fed_avg', 'fedlcc', 'fed_attn'], default='fedlcc')
    parser.add_argument('--num_users', type=int, default=10)
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument("--optimizer", type=str, default="Adam") 
    
    parser.add_argument("--nn_train_batch_size", default=20, type=int,
                        help="Batch size for training.")
    
    parser.add_argument("--local_epoch", type=int, default=1, help='the number of local epochs: E')
    parser.add_argument('--local_gradient_accumulation_steps', type=int, default=2)
    parser.add_argument("--local_train_batch_size", default=3, type=int,
                    help="Batch size for training.")

    parser.add_argument('--lr', type=float, default=0.000025)
    parser.add_argument("--l2_penalty", type=float, default=0.00000005)
    parser.add_argument("--lr_decay", type=float, default=0.01)


    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--test_batch_size", default=10, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--num_train_epochs", default=50.0, type=float,
                        help="Total number of training epochs to perform.")
    
    parser.add_argument('--moon_mu', type=float, default=0.1, help='coefficient of contrast loss')
    
    parser.add_argument("--stepsize", type=float, default=1, help='step size for aggregation')
    parser.add_argument('--metric', type=int, default=2, help='similarity metric')
    parser.add_argument("--dp", type=float, default=0.001, help='magnitude of randomization')
    
    args = parser.parse_args()
    return args
    
def main():
    args = get_args()
    device = torch.device("cuda", args.gpu_id)
    args.device = device
    
    save_path = (os.path.join(args.save_path + args.dataset_name))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    argument_path = os.path.join(save_path, '%s-num_users=%s-frac=%s') %(args.fed_algo, args.num_users, args.frac)
    # argument_path = os.path.join(save_path, '%s-lbs=%s-le=%s') %(args.fed_algo, args.local_train_batch_size, args.local_epoch)
    log=open(argument_path, mode='a')
    
    config = AutoConfig.from_pretrained(args.model_name_or_path,num_labels=args.num_class)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    args.dataset_dir = os.path.join(args.data_dir, args.dataset_name)
    
    train_file = os.path.join(args.dataset_dir, args.train_file)
    dev_file = os.path.join(args.dataset_dir, args.dev_file)
    test_file = os.path.join(args.dataset_dir, args.test_file)
    
    if os.path.exists(args.dataset_dir+"/process"):
        with open(args.dataset_dir+"/process/train_features.pkl", 'rb') as f:
            train_features = pickle.load(f)
        with open(args.dataset_dir+"/process/dev_features.pkl", 'rb') as f:
            dev_features = pickle.load(f)
        with open(args.dataset_dir+"/process/test_features.pkl", 'rb') as f:
            test_features = pickle.load(f)
            
    else:    
        os.makedirs(args.dataset_dir+"/process/")
        train_features = read_bio(args.dataset_name, train_file, tokenizer, max_seq_length=args.max_seq_length)
        dev_features = read_bio(args.dataset_name, dev_file, tokenizer, max_seq_length=args.max_seq_length)
        test_features = read_bio(args.dataset_name, test_file, tokenizer, max_seq_length=args.max_seq_length)
    
        with open(args.dataset_dir+'/process/train_features.pkl', 'wb') as f:
            pickle.dump(train_features, f)
        with open(args.dataset_dir+'/process/dev_features.pkl', 'wb') as f:
            pickle.dump(dev_features, f)
        with open(args.dataset_dir+'/process/test_features.pkl', 'wb') as f:
            pickle.dump(test_features, f)
            
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    model = Model(config, model, num_class=args.num_class)
    model.to(args.device)
    
    if args.fed_algo == 'None':
        nn_train(args, log, model, train_features, dev_features, test_features)
    else:
        print("RE in Federated learning !")
        fed_train(args, log, model, train_features, dev_features, test_features)
            
if __name__ == "__main__":
    main()