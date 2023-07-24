import argparse
import copy
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import trange
import csv

import sys

from experiments.FLHetero.Models.CNNs import CNN_1
from experiments.FLHetero.node import BaseNodes
from experiments.utils import get_device, set_logger, set_seed, str2bool

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
random.seed(2022)



def TestAcc(net, testloader,criteria):
    net.eval()
    with torch.no_grad():
        test_acc = 0
        test_loss = 0
        num_batch = 0

        for batch in testloader:
            num_batch += 1
            # batch = next(iter(testloader))
            img, label = tuple(t.to(device) for t in batch)

            pred, _ = net(img)
            test_loss += criteria(pred, label)
            test_acc += pred.argmax(1).eq(label).sum().item() / len(label)

        mean_test_loss = test_loss / num_batch
        mean_test_acc = test_acc / num_batch

    return mean_test_loss, mean_test_acc

def train(data_name: str, num_classes: int, data_path: str, classes_per_node: int, num_nodes: int, frac: float,
          steps: int, inner_steps: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int) -> None:

    ###############################
    # init nodes, hnet, local net #
    ###############################
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node,
                      batch_size=bs)

    # -------compute aggregation weights-------------#
    train_sample_count = nodes.train_sample_count
    eval_sample_count = nodes.eval_sample_count
    test_sample_count = nodes.test_sample_count

    client_sample_count = [train_sample_count[i] + eval_sample_count[i] + test_sample_count[i] for i in
                           range(len(train_sample_count))]

    # -----------------------------------------------#

    embed_dim = embed_dim
    if embed_dim == -1:
        logging.info("auto embedding size")


    if data_name == "cifar10":
        net = CNN_1(n_kernels=n_kernels)
    elif data_name == "cifar100":
        net = CNN_1(n_kernels=n_kernels, out_dim=100)
    elif data_name == "mnist":
        net = CNN_1(n_kernels=n_kernels)
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")



    net = net.to(device)
    init_Gnet_paras = copy.deepcopy(net.state_dict())

    ##################
    # init optimizer #
    ##################
    optimizers = {
        'sgd': torch.optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=wd),
        'adam': torch.optim.Adam(params=net.parameters(), lr=lr)
    }
    optimizer = optimizers[optim]
    criteria = torch.nn.CrossEntropyLoss()

    ################
    # init metrics #
    ################
    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(steps)
    results = defaultdict(list)

    last_global_acc = 0
    cur_global_acc = 0

    Share_keys = []
    Private_keys = []
    count = 0
    for k, v in init_Gnet_paras.items():
        if count < len(init_Gnet_paras.keys()) - 2:
            Private_keys.append(k) # extractor
            count += 1
        else:
            Share_keys.append((k)) # header

    extractor = dict([(k, init_Gnet_paras[k]) for k in Private_keys])
    header = dict([(k, init_Gnet_paras[k]) for k in Share_keys])

    Local_headers = defaultdict()
    Local_extractors = defaultdict()
    Local_Header = defaultdict()
    Alphas = defaultdict()

    PM_acc = defaultdict()
    Local_PMs = defaultdict()
    Owned_All_Classes = defaultdict()
    Train_data_distribution = defaultdict()


    Global_class_headers = defaultdict()
    for key, paras in header.items():
        Global_class_headers[key] = defaultdict()
        for s in range(num_classes):
            Global_class_headers[key][s] = header[key][s]

    for i in range(num_nodes):
        Local_extractors[i] = extractor
        Local_headers[i] = defaultdict()
        Local_PMs[i] = init_Gnet_paras
        Alphas[i] = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(device)
        Alphas[i].data.fill_(0.5)

        # Local_Header[i] = header
        PM_acc[i] = 0

        owned_all_classes = []
        Train_data_distribution[i] = defaultdict()
        for j, batch in enumerate(nodes.train_loaders[i], 0):
            img, label = tuple(t.to(device) for t in batch)
            classes = label.unique().detach().cpu().numpy()
            for s in classes:
                if s not in owned_all_classes:
                    owned_all_classes.append(s)
                cls_img = torch.stack(list(map(lambda x: x[0], filter(lambda x: x[1] == s, zip(img, label)))), 0)
                if j==0:
                    Train_data_distribution[i][s] = 0
                Train_data_distribution[i][s] += len(cls_img)
        Owned_All_Classes[i] = owned_all_classes


        for key, paras in header.items():
            Local_headers[i][key] = defaultdict()
            for cls in owned_all_classes:
                Local_headers[i][key][cls] = header[key][cls]

    Train_data_distribution_Class = defaultdict()
    for s in range(num_classes):
        Train_data_distribution_Class[s] = defaultdict()
        for id, cls_count in Train_data_distribution.items():
            for cls, count in cls_count.items():
                if cls == s:
                    Train_data_distribution_Class[s][id] = count

    decay_rounds = 45
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(str(save_path / f"Homo_FedSSA_DecayRound{decay_rounds}_R{steps}_N{num_nodes}_C{frac}_E{inner_steps}_NOIID_{data_name}_class_{classes_per_node}.csv"),'w', newline='') as file:

        mywriter = csv.writer(file, delimiter=',')

        for step in step_iter:
            frac = frac
            select_nodes = random.sample(range(num_nodes), int(frac * num_nodes))


            all_local_loss = []
            all_local_acc = []

            results = []

            logging.info(f'#-----------Round:{step}-------------#')
            for c in select_nodes:
                node_id = c
                round_id = step

                inner_optim = torch.optim.SGD(
                    net.parameters(), lr=inner_lr, momentum=.9, weight_decay=inner_wd
                )

                mse_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)


                for key, _ in header.items():
                    for cls in Owned_All_Classes[node_id]:
                        if step <= decay_rounds: # 10
                            alpha = 0.5 * np.cos(step*np.pi/(decay_rounds*2)) # 20
                        else:
                            alpha = 0

                        Local_PMs[node_id][key][cls] = Local_PMs[node_id][key][cls] * alpha + Global_class_headers[key][cls]



                net_paras = Local_PMs[node_id]
                net.load_state_dict(net_paras)



                net.train()
                # local training
                for i in range(inner_steps):
                    for j, batch in enumerate(nodes.train_loaders[node_id], 0):
                        inner_optim.zero_grad()
                        optimizer.zero_grad()

                        img, label = tuple(t.to(device) for t in batch)
                        pred, _ = net(img)

                        owned_classes = label.unique().detach().cpu().numpy()

                        loss = criteria(pred, label)
                        # print(f'Round:{round_id} | Client:{node_id} | key:{key} | cls:{cls} | hard_loss:{criteria(pred, label)} | dis:{Distance}')

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 50)

                        inner_optim.step()

                        local_model_dict = copy.deepcopy(net.state_dict())
                        Local_PMs[node_id] = local_model_dict
                        # Local_extractors[node_id] = dict([(k, local_model_dict[k]) for k in Private_keys])
                        local_header = dict([(k, local_model_dict[k]) for k in Share_keys])
                        #
                        for key, paras in header.items():
                            for cls in Owned_All_Classes[node_id]:
                                Local_headers[node_id][key][cls] = local_header[key][cls]



                with torch.no_grad():
                    net.eval()
                    test_loss, test_acc = TestAcc(net, nodes.test_loaders[node_id], criteria)


                all_local_loss.append(test_loss.cpu())
                all_local_acc.append(test_acc)
                PM_acc[node_id] = test_acc
                logging.info(f'round {step} | client {node_id} | acc:{PM_acc[node_id]}')




            mean_test_loss = round(np.mean(all_local_loss), 4)
            mean_test_acc = round(np.mean(all_local_acc), 4)

            global_var_acc = round(np.var(all_local_acc), 4)

            cur_global_acc = mean_test_acc

            results.append([mean_test_loss, mean_test_acc] + [round(i, 4) for i in PM_acc.values()])
            mywriter.writerows(results)
            file.flush()
            logging.info(f'Round:{step} | mean_local_loss:{mean_test_loss} | mean_local_acc:{mean_test_acc}')


            delta_acc = cur_global_acc - last_global_acc

            # Aggregation
            for key, values in header.items():
                for cls, _ in Global_class_headers[key].items():
                    cls_collect = []
                    for id, cls_paras in Local_headers.items():
                        for cls_, paras in Local_headers[id][key].items():
                            if cls_ == cls:
                                cls_collect.append(paras)
                    if cls_collect != []:
                        new_paras = Global_class_headers[key][cls] * 0
                        for paras in cls_collect:
                            new_paras += paras
                        Global_class_headers[key][cls] = new_paras / len(cls_collect)


            logging.info(f'Global Headers are updated after aggregation')


        logging.info('Results json have saved successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="FedSSA Homogeneous Experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100','mnist'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--num-classes", type=int, default=10, help="number of total classes")
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
    parser.add_argument("--num-nodes", type=int, default=50, help="number of simulated nodes")

    parser.add_argument("--frac", type=int, default=0.2, help="fraction")


    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--inner-steps", type=int, default=10, help="number of inner steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-5, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="Results/temp", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if args.data_name == 'cifar10':
        args.classes_per_node = 2
    elif args.data_name == 'cifar100':
        args.classes_per_node = 10
    else:
        args.classes_per_node = 2

    train(
        data_name=args.data_name,
        num_classes=args.num_classes,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        num_nodes=args.num_nodes,
        frac=args.frac,
        steps=args.num_steps,
        inner_steps=args.inner_steps,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed
    )
