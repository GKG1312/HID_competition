import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import json
from torch.utils.data import Dataset, DataLoader
import cv2
from einops import rearrange
import math
from torch.amp import autocast, GradScaler

# Utility Functions
def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

def np2var(x, requires_grad=False, device='cuda'):
    tensor = torch.from_numpy(x).to(device)
    tensor.requires_grad = requires_grad
    return tensor

def list2var(x, device='cuda'):
    return torch.tensor(x, dtype=torch.long, device=device)

def ts2np(x):
    return x.cpu().numpy()

def mkdir(path):
    os.makedirs(path, exist_ok=True)

# Custom Collate Function
def custom_collate_fn(batch, max_seq_len=60):
    sils_list, labels, seq_info = [], [], []
    seq_lengths = []
    
    # Unpack batch
    for sils, label, info in batch:
        sils_list.append(sils)
        labels.append(label)
        seq_info.append(info)
        seq_lengths.append(sils.shape[0])
    
    # Pad or truncate sequences and adjust seq_lengths
    padded_sils = []
    adjusted_seq_lengths = []
    for sils, length in zip(sils_list, seq_lengths):
        curr_len = min(length, max_seq_len)
        if length > max_seq_len:
            sils = sils[:max_seq_len]
        elif length < max_seq_len:
            pad_len = max_seq_len - length
            sils = np.pad(sils, ((0, pad_len), (0, 0), (0, 0)), mode='constant', constant_values=0)
        padded_sils.append(sils)
        adjusted_seq_lengths.append(curr_len)
    
    # Stack into batch and convert to float32
    sils_batch = torch.stack([torch.from_numpy(s).float() for s in padded_sils], dim=0)
    labels_batch = torch.tensor(labels, dtype=torch.long)
    seq_lengths = torch.tensor(adjusted_seq_lengths, dtype=torch.int)
    
    return sils_batch, labels_batch, seq_info, seq_lengths

# Dataset
class HIDDataset(Dataset):
    def __init__(self, dataset_root, partition_file, training=True, transform=None):
        self.dataset_root = dataset_root
        self.transform = transform
        with open(partition_file, 'r') as f:
            partition = json.load(f)
        label_list = os.listdir(dataset_root)
        train_set = [label for label in partition.get('TRAIN_SET', []) if label in label_list]
        test_set = [label for label in partition.get('TEST_SET', []) if label in label_list]
        self.seqs_info = self._get_seqs_info(training and train_set or test_set)
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.label_set = sorted(list(set(self.label_list)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_set)}

    def _get_seqs_info(self, label_set):
        seqs_info_list = []
        for lab in label_set:
            for typ in sorted(os.listdir(os.path.join(self.dataset_root, lab))):
                for vie in sorted(os.listdir(os.path.join(self.dataset_root, lab, typ))):
                    seq_info = [lab, typ, vie]
                    seq_path = os.path.join(self.dataset_root, *seq_info)
                    seq_dirs = sorted(os.listdir(seq_path))
                    if seq_dirs:
                        seq_dirs = [os.path.join(seq_path, dir) for dir in seq_dirs]
                        seqs_info_list.append([*seq_info, seq_dirs])
        return seqs_info_list

    def __len__(self):
        return len(self.seqs_info)

    def __getitem__(self, idx):
        seq_info = self.seqs_info[idx]
        paths = sorted(seq_info[-1])
        if not paths:
            raise ValueError(f"No .pkl files found for sequence {seq_info}")
        pth = paths[0]
        if not pth.endswith('.pkl'):
            raise ValueError(f"Unsupported file: {pth}")
        with open(pth, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, np.ndarray):
            sils = data
        elif isinstance(data, dict) and 'silhouettes' in data:
            sils = data['silhouettes']
        else:
            raise ValueError(f"Unexpected data format in {pth}: {type(data)}")
        if not isinstance(sils, np.ndarray) or sils.ndim != 3:
            raise ValueError(f"Invalid silhouette array in {pth}: shape {np.shape(sils)}")
        if sils.shape[1:] != (128, 128):
            raise ValueError(f"Expected 128x128 silhouettes in {pth}, got shape {sils.shape}")
        if self.transform:
            sils = self.transform(sils)
        # print(f"Sequence {seq_info[0]}: sils shape after transform: {sils.shape}")   #DEBUG
        if sils.shape[0] < 1:
            raise ValueError(f"Empty sequence for {seq_info}")
        label = self.label_to_idx[seq_info[0]]
        return sils, label, seq_info

# Transform
class BaseSilCuttingTransform:
    def __init__(self, divisor=255.0, img_w=64, img_h=64):
        self.divisor = divisor
        self.img_w = img_w
        self.img_h = img_h

    def __call__(self, x):
        # print(f"Transform input shape: {x.shape}")  #DEBUG
        if x.shape[1:] != (self.img_h, self.img_w):
            x = np.stack([cv2.resize(frame, (self.img_w, self.img_h)) for frame in x], axis=0)
        return (x / self.divisor).astype(np.float32)

# Modules
class HorizontalPoolingPyramid(nn.Module):
    def __init__(self, bin_num=None):
        super().__init__()
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def forward(self, x):
        n, c = x.size()[:2]
        # print(f"HPP: input shape {x.shape}, bin_num {self.bin_num}")  #DEBUG 
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)

class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super().__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        n, c, s, h, w = x.size()
        x = self.forward_block(x.transpose(1, 2).reshape(-1, c, h, w), *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()

class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super().__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        # print(f"PackSequenceWrapper: seqL {seqL}, seqs shape {seqs.shape}")  # DEBUG
        batch_size = seqs.size(0)
        rets = []
        for i in range(batch_size):
            curr_seqL = min(seqL[i], seqs.size(dim))
            if curr_seqL > 0:
                narrowed_seq = seqs[i : i + 1].narrow(dim, 0, curr_seqL)
                rets.append(self.pooling_func(narrowed_seq, **options))
            else:
                rets.append(torch.zeros_like(self.pooling_func(seqs[i : i + 1].narrow(dim, 0, 1), **options)))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets]) for j in range(len(rets[0]))]
        return torch.cat(rets)

class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super().__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()

class SeparateBNNecks(nn.Module):
    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super().__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = nn.ModuleList([nn.BatchNorm1d(in_channels) for _ in range(parts_num)])
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x.squeeze(2)).unsqueeze(2) for _x, bn in zip(x.split(1, 2), self.bn1d)], 2)
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)
            logits = feature.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()

# Loss Functions
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, loss_term_weight=1.0):
        super().__init__()
        self.margin = margin
        self.loss_term_weight = loss_term_weight

    def forward(self, embeddings, labels):
        embeddings = embeddings.permute(2, 0, 1).float()
        dist = self.compute_distance(embeddings, embeddings)
        ap_dist, an_dist = self.convert_to_triplets(labels, labels, dist)
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)
        loss_avg = loss.mean()
        return loss_avg, {'loss': loss_avg.item()}

    def compute_distance(self, x, y):
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)
        inner = x.matmul(y.transpose(1, 2))
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))
        return dist

    def convert_to_triplets(self, row_labels, col_labels, dist):
        matches = (row_labels.unsqueeze(1) == col_labels.unsqueeze(0)).bool()
        diffenc = torch.logical_not(matches)
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist

class CrossEntropyLoss(nn.Module):
    def __init__(self, scale=16, label_smooth=True, eps=0.1, loss_term_weight=0.1):
        super().__init__()
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.loss_term_weight = loss_term_weight

    def forward(self, logits, labels):
        n, c, p = logits.size()
        logits = logits.float()
        labels = labels.unsqueeze(1)
        if self.label_smooth:
            loss = F.cross_entropy(logits * self.scale, labels.repeat(1, p), label_smoothing=self.eps)
        else:
            loss = F.cross_entropy(logits * self.scale, labels.repeat(1, p))
        return loss, {'loss': loss.item()}

# Loss Aggregator
class LossAggregator(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.losses = nn.ModuleDict({
            cfg['log_prefix']: self._build_loss(cfg) for cfg in loss_cfg
        })

    def _build_loss(self, loss_cfg):
        if loss_cfg['type'] == 'Triplet':
            params = {k: loss_cfg[k] for k in ['margin', 'loss_term_weight'] if k in loss_cfg}
            return TripletLoss(**params)
        elif loss_cfg['type'] == 'CrossEntropy':
            params = {k: loss_cfg[k] for k in ['scale', 'label_smooth', 'eps', 'loss_term_weight'] if k in loss_cfg}
            return CrossEntropyLoss(**params)
        else:
            raise ValueError(f"Unsupported loss type: {loss_cfg['type']}")

    def forward(self, training_feats):
        loss_sum = 0.0
        loss_info = {}
        for k, v in training_feats.items():
            if k in self.losses:
                loss_func = self.losses[k]
                loss, info = loss_func(**v)
                for name, value in info.items():
                    loss_info[f'scalar/{k}/{name}'] = value
                loss = loss.mean() * loss_func.loss_term_weight
                loss_sum += loss
            else:
                raise ValueError(f"Unknown feature key: {k}")
        return loss_sum, loss_info

# Backbone
class PlainBackbone(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)

# GaitBase Model
class GaitBase(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.backbone = PlainBackbone(model_cfg['backbone_cfg']['in_channels'])
        self.fcs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.bn_necks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.hpp = HorizontalPoolingPyramid(model_cfg['bin_num'])
        self.temporal_pool = PackSequenceWrapper(torch.max)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        if not isinstance(ipts, (list, tuple)):
            ipts = [ipts]
        sils = ipts[0]
        # print(f"Input sils shape: {sils.shape}, seqL: {seqL}")  #DEBUG
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected sils shape: {sils.shape}")
        # print(f"Reshaped sils shape: {sils.shape}")  #DEBUG
        outs = self.backbone(sils)
        outs = self.temporal_pool(outs, seqL, options={'dim': 2})[0]
        feat = self.hpp(outs)
        # print(f"After HPP: feat shape {feat.shape}")  #DEBUG
        embed_1 = self.fcs(feat)
        embed_2, logits = self.bn_necks(embed_1)
        training_feat = {
            'triplet': {'embeddings': embed_1, 'labels': labs},
            'softmax': {'logits': logits, 'labels': labs}
        }
        return {'training_feat': training_feat, 'inference_feat': {'embeddings': embed_1}}

# Training Function
def train(model, dataloader, optimizer, loss_aggregator, device, epoch):
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler('cuda')
    for batch_idx, (sils, labels, _, seq_lengths) in enumerate(dataloader):
        sils, labels, seq_lengths = sils.to(device), labels.to(device), seq_lengths.to(device)
        # print(f"Batch {batch_idx}: sils shape {sils.shape}, seq_lengths {seq_lengths}, sils dtype {sils.dtype}, max_seq_len 60")  DEBUG
        if (seq_lengths > 60).any():
            raise ValueError(f"seq_lengths {seq_lengths} exceed max_seq_len 60")
        inputs = ([sils], labels, None, None, [seq_lengths])
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss_sum, loss_info = loss_aggregator(outputs['training_feat'])
        scaler.scale(loss_sum).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss_sum.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss_sum.item():.4f}, {loss_info}")
    return total_loss / len(dataloader)

# Main
def main():
    # Configuration
    dataset_root = r"D:\personalProject\hid_project\gallery"
    partition_file = r"code\gaitbase\HID.json"
    checkpoint_path = r"D:\personalProject\OpenGait\output\HID\Baseline\Baseline\checkpoints\Baseline-60000.pt"
    output_dir = r"output\HID\Baseline_HID_finetuned"
    loss_cfg = [
        {'type': 'Triplet', 'log_prefix': 'triplet', 'margin': 0.2, 'loss_term_weight': 1.0},
        {'type': 'CrossEntropy', 'log_prefix': 'softmax', 'scale': 16, 'label_smooth': True, 'eps': 0.1, 'loss_term_weight': 0.1}
    ]
    lr = 0.01
    total_epochs = 50
    batch_size = 8
    save_iter = 10

    # Verify number of classes
    def verify_classes(dataset_root, partition_file):
        with open(partition_file, 'r') as f:
            partition = json.load(f)
        label_list = os.listdir(dataset_root)
        train_set = [label for label in partition.get('TRAIN_SET', []) if label in label_list]
        num_classes = len(set(train_set))
        print(f"Number of classes in training set: {num_classes}")
        return num_classes

    num_classes = verify_classes(dataset_root, partition_file)
    model_cfg = {
        'model': 'Baseline',
        'backbone_cfg': {'in_channels': 1, 'type': 'Plain'},
        'SeparateFCs': {'parts_num': 31, 'in_channels': 256, 'out_channels': 256, 'norm': False},
        'SeparateBNNecks': {'parts_num': 31, 'in_channels': 256, 'class_num': num_classes, 'norm': True, 'parallel_BN1d': True},
        'bin_num': [16, 8, 4, 2, 1]
    }

    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mkdir(output_dir)

    # Dataset and DataLoader
    transform = BaseSilCuttingTransform(img_w=64, img_h=64)
    dataset = HIDDataset(dataset_root, partition_file, training=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lambda b: custom_collate_fn(b, max_seq_len=60))

    # Model
    model = GaitBase(model_cfg).to(device)
    # In main
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model']
        state_dict.pop('bn_necks.fc_bin', None)
        model.load_state_dict(state_dict, strict=False)
        # Reinitialize bn_necks.fc_bin on the correct device
        with torch.no_grad():
            model.bn_necks.fc_bin = nn.Parameter(
                torch.randn(31, 256, num_classes, device=device)
            )
            nn.init.xavier_uniform_(model.bn_necks.fc_bin)
        print(f"Loaded and adapted checkpoint: {checkpoint_path}")
        # Verify all parameters are on the same device
        for name, param in model.named_parameters():
            # print(f"Parameter {name}: device {param.device}, dtype {param.dtype}")   #DEBUG
            if param.device != device:
                raise ValueError(f"Parameter {name} is on {param.device}, expected {device}")
            break
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Loss and Optimizer
    loss_aggregator = LossAggregator(loss_cfg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    # Training Loop
    for epoch in range(total_epochs):
        avg_loss = train(model, dataloader, optimizer, loss_aggregator, device, epoch)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        if (epoch + 1) % save_iter == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch},
                       os.path.join(output_dir, f"Baseline_HID_finetuned-{epoch+1}.pt"))

if __name__ == "__main__":
    main()
