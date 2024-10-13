import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import shelve


class AvgPoolingCnn(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.conv = nn.Sequential (
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
        )

        self.fc1 = nn.Linear(8112, 1024)

        self.out_v1 = nn.Linear(1024, 3)
        self.out_v2 = nn.Linear(1024, 3)
        self.out_v3 = nn.Linear(1024, 3)
        self.out_v4 = nn.Linear(1024, 3)
        self.out_v5 = nn.Linear(1024, 3)
    
    def forward(self, batch):
        for k,v in batch.items():
            batch[k] = v.permute(1, 0, 2, 3)

        series_1 = [t.reshape(-1, 1, 224, 224) for t in batch['series-1']]
        series_2 = [t.reshape(-1, 1, 224, 224) for t in batch['series-2']]
        series_3 = [t.reshape(-1, 1, 224, 224) for t in batch['series-3']]

        features_1 = []
        features_2 = []
        features_3 = []
        features_all = []

        zipped_iter = zip(
            [series_1, series_2, series_3],
            [features_1, features_2, features_3],
        )

        for series, features in zipped_iter:
            for t in series:
                # t.size() = torch.Size([B, 1, 224, 224])
                t = self.conv(t)
                t = torch.flatten(t, start_dim=1)

                features.append(t)
            tsr = torch.stack(features, dim=1)
            # tsr.size() = torch.Size([14, X, 2704])
            features_all.append(tsr.mean(1))

        features = torch.cat(features_all, dim=1)
        # features.size() = torch.Size([B, 8112])
        features = self.fc1(features)
        features = torch.relu(features)

        out1 = self.out_v1(features)
        out2 = self.out_v2(features)
        out3 = self.out_v3(features)
        out4 = self.out_v4(features)
        out5 = self.out_v5(features)

        return torch.stack((out1, out2, out3, out4, out5), dim=1)
    

class ShelveDataset(Dataset):
    def __init__(self, shelve_path):
        self.shelve_path = shelve_path
        with shelve.open(self.shelve_path) as db:
            self.keys = list(db.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        with shelve.open(self.shelve_path) as db:
            sample = db[self.keys[idx]]
        
        return sample["X"], sample["Y"]

    @staticmethod
    def collate_fn(batch):
        features, labels = zip(*batch)
        batch_size = len(features)

        # Pad features
        series_max_len = {f'series-{i}':0 for i in range(1, 4)}
        for f in features:
            for s in series_max_len:
                l = len(f[s])
                if l > series_max_len[s]:
                    series_max_len[s] = l
        
        for f in features:
            for s, l in series_max_len.items():
                for i in range(len(f[s])):
                    f[s][i] = torch.Tensor(f[s][i])
                f[s] += [torch.zeros((224, 224)) for _ in range(len(f[s]), l)]

        for label in labels:
            for k, v in label.items():
                label[k] = torch.tensor(v, dtype=torch.float32)
        
        # construct feature batches
        features_batch = {
            "series-1": [],
            "series-2": [],
            "series-3": [],
        }

        for f in features:
            for series in series_max_len:
                features_batch[series].append(torch.stack(f[series]))

        for series in series_max_len:
            features_batch[series] = torch.stack(features_batch[series])

        # construct label batches
        labels_batch = {k:[] for k in labels[0].keys()}

        for label in labels:
            for condition in labels_batch.keys():
                labels_batch[condition].append(label[condition])

        for condition in labels_batch:
            labels_batch[condition] = torch.stack(labels_batch[condition])

        return features_batch, labels_batch
