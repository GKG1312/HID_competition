import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
from einops import rearrange

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

# Modules
class HorizontalPoolingPyramid(nn.Module):
    def __init__(self, bin_num=None):
        super().__init__()
        if bin_num is None:
            bin_num = [16]
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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BasicBlockP3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer2d = nn.BatchNorm2d
            norm_layer3d = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(inplanes, planes, stride),
                norm_layer2d(planes),
                nn.ReLU(inplace=True)
            )
        )
        self.conv2 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(planes, planes),
                norm_layer2d(planes),
            )
        )
        self.shortcut3d = nn.Conv3d(planes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False)
        self.sbn = norm_layer3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out + self.sbn(self.shortcut3d(out)))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=[1, 1, 1], downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        assert stride[0] in [1, 2, 3]
        if stride[0] in [1, 2]:
            tp = 1
        else:
            tp = 0
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=[tp, 1, 1], bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=[1, 1, 1], bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# DeepGaitV2 Model
class DeepGaitV2(nn.Module):
    blocks_map = {
        '2d': BasicBlock2D,
        'p3d': BasicBlockP3D,
        '3d': BasicBlock3D
    }

    def __init__(self, model_cfg):
        super().__init__()
        mode = model_cfg['Backbone']['mode']
        assert mode in self.blocks_map.keys()
        block = self.blocks_map[mode]
        in_channels = model_cfg['Backbone']['in_channels']
        layers = model_cfg['Backbone']['layers']
        channels = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg.get('use_emb2', False)
        strides = [
            [1, 1],
            [2, 2],
            [2, 2],
            [1, 1]
        ]
        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))
        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)
        if mode == '2d':
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)
        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):
        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion)
                )
        else:
            downsample = lambda x: x
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(block(self.inplanes, planes, stride=s))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88, 128]
        del ipts
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        outs = self.TP(out4, seqL, options={"dim": 2})[0]
        feat = self.HPP(outs)
        embed_1 = self.FCs(feat)
        embed_2, logits = self.BNNecks(embed_1)
        if self.inference_use_emb2:
            embed = embed_2
        else:
            embed = embed_1
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval

# Custom Collate Function
def custom_collate_fn(batch, max_seq_len=60):
    sils_list, video_ids = [], []
    seq_lengths = []
    for sils, video_id in batch:
        sils_list.append(sils)
        video_ids.append(video_id)
        seq_lengths.append(sils.shape[0])
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
    def __init__(self, divisor=255.0, img_w=128, img_h=128):
        self.divisor = divisor
        self.img_w = img_w
        self.img_h = img_h

    def __call__(self, x):
        if x.shape[1:] != (self.img_h, self.img_w):
            x = np.stack([cv2.resize(frame, (self.img_w, self.img_h)) for frame in x], axis=0)
        return (x / self.divisor).astype(np.float32)

# Inference Function
def infer(model, dataloader, device, output_csv):
    model.eval()
    results = []
    with torch.no_grad():
        for sils, video_ids, seq_lengths in dataloader:
            sils, seq_lengths = sils.to(device), seq_lengths.to(device)
            dummy_labels = torch.zeros(sils.size(0), dtype=torch.long, device=device)
            inputs = ([sils], dummy_labels, None, None, [seq_lengths])
            outputs = model(inputs)
            logits = outputs['training_feat']['softmax']['logits']
            logits = logits.mean(dim=2)
            predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
            for video_id, pred_class in zip(video_ids, predicted_classes):
                results.append({
                    'videoID': os.path.splitext(video_id)[0],
                    'label': int(pred_class)
                })
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved inference results to {output_csv}, Entries: {len(df)}")

# Main
def main():
    # Configuration
    dataset_root = r"D:\personalProject\hid_project\probe_phase2"
    checkpoint_path = r"D:\personalProject\hid_project\output\HID\Baseline_HID_finetuned\DeepGaitV2-60000.pt"
    output_csv = r"output\HID\DeepGaitV2_HID_finetuned\submission.csv"
    model_cfg = {
        'model': 'DeepGaitV2',
        'Backbone': {
            'mode': 'p3d',
            'in_channels': 1,
            'layers': [1, 1, 1, 1],
            'channels': [64, 128, 256, 512]
        },
        'SeparateBNNecks': {
            'parts_num': 16,
            'in_channels': 256,
            'class_num': 859,
            'norm': True,
            'parallel_BN1d': True
        },
        'use_emb2': True
    }
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mkdir(os.path.dirname(output_csv))
    transform = BaseSilCuttingTransform(img_w=128, img_h=128)
    dataset = UnlabeledHIDDataset(dataset_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=custom_collate_fn)
    model = DeepGaitV2(model_cfg).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    infer(model, dataloader, device, output_csv)

if __name__ == "__main__":
    main()
