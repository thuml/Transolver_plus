import torch
import os
import random
import numpy as np
import h5py
import json

class AirplaneDataset(torch.utils.data.Dataset):
    def __init__(self, path, train=True, train_set=None, split_size=600000):
        self.split_size = split_size
        filename = '/aircraft_docker/airplane_dataset.json'
        with open(filename, 'r') as f:
            data = json.load(f)
        if train:
            self.train_set = data['train_set']
        else:
            self.train_set = data['test_set']
        if train_set is not None:
            self.train_set = train_set

        self.f_list = [os.path.join(path, f) for f in self.train_set]
        self.num_points_list = []
        for f in self.f_list:
            with h5py.File(f, "r") as h5:
                num_points = h5["pos"].shape[0]
            self.num_points_list.append(num_points)

    def __len__(self):
        return len(self.f_list)
    
    def __getitem__(self, idx):
        return self.f_list[idx], self.num_points_list[idx]

class AirplaneDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, sampler=None):
        super(AirplaneDataLoader, self).__init__(dataset, batch_size=batch_size, sampler=sampler)
        self.split_size = dataset.split_size
    
    def __iter__(self):
        for idx in self.sampler:
            file, num_points = self.dataset[idx]
            start = 0
            end = num_points
            with h5py.File(file, "r") as f:
                pos = torch.from_numpy(f["pos"][start:end]).unsqueeze(0).float()
                normals = torch.from_numpy(f["normals"][start:end]).unsqueeze(0).float()
                values = torch.from_numpy(f["values"][start:end]).unsqueeze(0).float()
                # attributes
                mach = torch.tensor(f.attrs["Ma"]).unsqueeze(0).float()
                alpha = torch.tensor(f.attrs["alpha"]).unsqueeze(0).float()
                beta = torch.tensor(f.attrs["beta"]).unsqueeze(0).float()
                
            sdf = torch.zeros((1, num_points, 1))
            x = torch.cat([pos, sdf, normals], dim=-1)
            condition = torch.cat([mach, alpha, beta], dim=-1).unsqueeze(0)
            yield x, values, pos, condition, None