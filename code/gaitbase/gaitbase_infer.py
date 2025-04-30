import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2

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
    sils_list, video_ids = [], []
    seq_lengths = []
    
    # Unpack batch
    for sils, video_id in batch:
        sils_list.append(sils)
        video_ids.append(video_id)
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
    seq_lengths = torch.tensor(adjusted_seq_lengths, dtype=torch.int)
    
    return sils_batch, video_ids, seq_lengths

# Dataset
class UnlabeledHIDDataset(Dataset):
    def __init__(self, dataset_root, transform=None):
        self.dataset_root = dataset_root
        self.transform = transform
        self.seqs_info = self._get_seqs_info()

    def _get_seqs_info(self):
        seqs_info_list = []
        for root, _, files in os.walk(self.dataset_root):
            for file in sorted(files):
                if file.endswith('.pkl'):
                    seq_path = os.path.join(root, file)
                    seqs_info_list.append([seq_path, file])
        return seqs_info_list

    def __len__(self):
        return len(self.seqs_info)

    def __getitem__(self, idx):
        seq_path, video_id = self.seqs_info[idx]
        with open(seq_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, np.ndarray):
            sils = data
        elif isinstance(data, dict) and 'silhouettes' in data:
            sils = data['silhouettes']
        else:
            raise ValueError(f"Unexpected data format in {seq_path}: {type(data)}")
        if not isinstance(sils, np.ndarray) or sils.ndim != 3:
            raise ValueError(f"Invalid silhouette array in {seq_path}: shape {np.shape(sils)}")
        if sils.shape[1:] != (128, 128):
            raise ValueError(f"Expected 128x128 silhouettes in {seq_path}, got shape {sils.shape}")
        if self.transform:
            sils = self.transform(sils)
        if sils.shape[0] < 1:
            raise ValueError(f"Empty sequence in {seq_path}")
        return sils, video_id

# Transform
class BaseSilCuttingTransform:
    def __init__(self, divisor=255.0, img_w=64, img_h=64):
        self.divisor = divisor
        self.img_w = img_w
        self.img_h = img_h

    def __call__(self, x):
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
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected sils shape: {sils.shape}")
        outs = self.backbone(sils)
        outs = self.temporal_pool(outs, seqL, options={'dim': 2})[0]
        feat = self.hpp(outs)
        embed_1 = self.fcs(feat)
        embed_2, logits = self.bn_necks(embed_1)
        training_feat = {
            'triplet': {'embeddings': embed_1, 'labels': labs},
            'softmax': {'logits': logits, 'labels': labs}
        }
        return {'training_feat': training_feat, 'inference_feat': {'embeddings': embed_1, 'logits': logits}}

# Inference Function
def infer(model, dataloader, device, output_csv):
    model.eval()
    results = []
    with torch.no_grad():
        for sils, video_ids, seq_lengths in dataloader:
            sils, seq_lengths = sils.to(device), seq_lengths.to(device)
            # Dummy labels (not used for inference)
            dummy_labels = torch.zeros(sils.size(0), dtype=torch.long, device=device)
            inputs = ([sils], dummy_labels, None, None, [seq_lengths])
            outputs = model(inputs)
            logits = outputs['training_feat']['softmax']['logits']  # Shape: [batch, 859, 31]
            # Average logits over parts
            logits = logits.mean(dim=2)  # Shape: [batch, 859]
            predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()  # Shape: [batch]
            for video_id, pred_class in zip(video_ids, predicted_classes):
                results.append({
                    'videoID': video_id,
                    'predicted_class': int(pred_class)
                })
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved inference results to {output_csv}, Entries: {len(df)}")

# Main
def main():
    # Configuration
    dataset_root = r"D:\personalProject\hid_project\probe_phase1"
    checkpoint_path = r"output\HID\Baseline_HID_finetuned\Baseline_HID_finetuned-50.pt"
    output_csv = r"output\HID\Baseline_HID_finetuned\submission.csv"
    model_cfg = {
        'model': 'Baseline',
        'backbone_cfg': {'in_channels': 1, 'type': 'Plain'},
        'SeparateFCs': {'parts_num': 31, 'in_channels': 256, 'out_channels': 256, 'norm': False},
        'SeparateBNNecks': {'parts_num': 31, 'in_channels': 256, 'class_num': 859, 'norm': True, 'parallel_BN1d': True},
        'bin_num': [16, 8, 4, 2, 1]
    }
    batch_size = 8

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mkdir(os.path.dirname(output_csv))

    # Dataset and DataLoader
    transform = BaseSilCuttingTransform(img_w=64, img_h=64)
    dataset = UnlabeledHIDDataset(dataset_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, 
                           collate_fn=lambda b: custom_collate_fn(b, max_seq_len=60))

    # Model
    model = GaitBase(model_cfg).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Inference
    infer(model, dataloader, device, output_csv)

if __name__ == "__main__":
    main()