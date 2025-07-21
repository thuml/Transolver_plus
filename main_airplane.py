import train_airplane as train
import os
import torch
import argparse
from torch.utils.data import RandomSampler
import logging
from dataset.dataset import AirplaneDataLoader, AirplaneDataset
import torch.distributed as dist
import datetime
import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/airplane_data/')
parser.add_argument('--save_dir', default='/data/airplane_data/')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--val_iter', default=10, type=int)
parser.add_argument('--cfd_config_dir', default='cfd/cfd_params.yaml')
parser.add_argument('--cfd_model')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--nb_epochs', default=200, type=int)
parser.add_argument('--preprocessed', default=1, type=int)
parser.add_argument('--finetune', default=0, type=int)
# add arguments related to normalization
parser.add_argument('--pos_norm', default=1, type=int)
parser.add_argument('--out_norm', default=1, type=int)
parser.add_argument('--dataset', default='drivernet')
parser.add_argument('--eval', default=False, type=bool)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--out-dim', default=4, type=int)
args = parser.parse_args()
print(args)

hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}

ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
port = os.environ.get("MASTER_PORT", "64209")
hosts = int(os.environ.get("WORLD_SIZE", "8"))  # number of nodes
rank = int(os.environ.get("RANK", "0"))  # node id
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
gpus = torch.cuda.device_count()  # gpus per node
args.local_rank = local_rank

dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                        rank=rank, timeout=datetime.timedelta(seconds=100))
torch.cuda.set_device(rank)

device = torch.device("cuda", rank)

train_dataset = AirplaneDataset(args.save_dir, train=True)
val_dataset = AirplaneDataset(args.save_dir, train=False)

train_sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(0))
val_sampler = RandomSampler(val_dataset, generator=torch.Generator().manual_seed(0))

train_loader = AirplaneDataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
val_loader = AirplaneDataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

# 200 case
pos_mean = torch.tensor([2.80879162e+03, 1.00957077e+02, 6.76594237e-03]).view(1, 1, 3).cuda()
pos_std = torch.tensor([1436.65326859, 178.37956359, 615.16521715]).view(1, 1, 3).cuda()
norm_mean = torch.tensor([-7.03865828e-02, 1.50757955e-01, -6.07368549e-06]).view(1, 1, 3).cuda()
norm_std = torch.tensor([0.19895465, 0.87515866, 0.40665163]).view(1, 1, 3).cuda()
out_mean = torch.tensor([0.04602036, 1.3157164, 5.66693757, 0.25599, 0.06231503, 1.64027649]).view(1, 1, 6).cuda()
out_std = torch.tensor([0.09458788, 0.76978003, 0.41717544, 0.47068753, 0.6710297, 1.8059161]).view(1, 1, 6).cuda()

from models.Transolver_plus import Model
model = Model(n_hidden=256, n_layers=4, space_dim=7,
                fun_dim=0,
                n_head=8,
                mlp_ratio=2, out_dim=6,
                slice_num=32,
                unified_pos=0,
                dropout=0.1).cuda()
# default
# path = f'metrics/airplane/{args.cfd_model}/{args.dataset}/{args.fold_id}/{args.nb_epochs}_{args.weight}'

path = "./"

if not os.path.exists(path):
    os.makedirs(path)

if args.eval:
    logging.basicConfig(filename=os.path.join(path, 'test.log'), level=logging.INFO, filemode='w', format='%(asctime)s - %(message)s')
    logging.info(args)
else:
    logging.basicConfig(filename=os.path.join(path, 'train.log'), level=logging.INFO, filemode='w', format='%(asctime)s - %(message)s')
    logging.info(args)

logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
logging.info(model)
print(model)

if not args.eval:
    # train
    model = train.main(device, train_loader, val_loader, model, hparams, path, val_iter=args.val_iter, reg=args.weight, pos_norm=args.pos_norm, out_norm=args.out_norm, norm_norm=0, pos_mean=pos_mean, pos_std=pos_std, out_mean=out_mean, out_std=out_std, norm_mean=norm_mean, norm_std=norm_std, full=True)
else:
    data_dir_h5 = '/aircraft_data'
    res_file = '/aircraft_data/result.csv'
    df = pd.read_csv(res_file)
    id = 0

    pos_mean = torch.tensor([2.80879162e+03, 1.00957077e+02, 6.76594237e-03]).view(1, 1, 3).cuda()
    pos_std = torch.tensor([1436.65326859, 178.37956359, 615.16521715]).view(1, 1, 3).cuda()
    norm_mean = torch.tensor([-7.03865828e-02, 1.50757955e-01, -6.07368549e-06]).view(1, 1, 3).cuda()
    norm_std = torch.tensor([0.19895465, 0.87515866, 0.40665163]).view(1, 1, 3).cuda()
    out_mean = torch.tensor([0.04602036, 1.3157164, 5.66693757, 0.25599, 0.06231503, 1.64027649]).view(1, 1, 6).cuda()
    out_std = torch.tensor([0.09458788, 0.76978003, 0.41717544, 0.47068753, 0.6710297, 1.8059161]).view(1, 1, 6).cuda()

    model = torch.load("./model_200.pth").cuda()
    l2re = 0
    for index, row in df.iloc[-14:].iterrows():
        idx = row['idx']
        Ma = row['Ma']
        alpha = row['alpha']
        beta = row['beta']
        in_file_h5 = os.path.join(data_dir_h5, f'{int(idx)}_{Ma}_{alpha}_{beta}.h5')

        with h5py.File(in_file_h5, 'r') as f:
            normals = f['normals'][:]
            pos = f['pos'][:]
            values = f['values'][:]
    
        with torch.no_grad():
            pos = torch.tensor(pos, dtype=torch.float32).view(1, -1, 3).cuda()
            normals = torch.tensor(normals, dtype=torch.float32).view(1, -1, 3).cuda()
            pos = (pos - pos_mean) / pos_std
            N = pos.shape[1]
            x = torch.cat([pos, torch.zeros((1, N, 1), dtype=torch.float32).cuda(), normals], dim=2)
            condition = torch.tensor([Ma, alpha, beta]).view(1, 3).cuda().float()
            out = model((x, pos, condition))
            out = out * out_std + out_mean
            out = out.cpu().numpy()
            l2re += np.linalg.norm(out[0, :, -1] - values[:, -1]) / np.linalg.norm(values[:, -1])
            # save output with name
            np.save(f"output/{idx}_{Ma}_{alpha}_{beta}.npy", out)
        id += 1

    print(f"Average L2RE: {l2re / id}")
        
