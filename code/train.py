# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# from torch.amp import GradScaler, autocast
# import os
# import numpy as np
# from dataloader import get_dataloaders, backbone_configs
# from classifier import GaitClassifier
# from losses import ArcFace

# def get_backbone(backbone_type, hidden_dim=512, frame_count=40):
#     """
#     Factory function to instantiate backbone based on type.
#     Args:
#         backbone_type (str): Backbone name (e.g., 'gaitstar', 'gaitset', 'iresnet50').
#         hidden_dim (int): Embedding dimension.
#         frame_count (int): Number of frames.
#     Returns:
#         nn.Module: Backbone instance.
#     """
#     if backbone_type == 'gaitstar':
#         from backbones.gaitstar import GaitSTARBackbone
#         return GaitSTARBackbone(hidden_dim=hidden_dim, frame_count=frame_count)
#     elif backbone_type == 'gaitset':
#         from backbones.GaitSet_cnn import GaitSetBackbone
#         return GaitSetBackbone(hidden_dim=hidden_dim)
#     elif backbone_type == 'vgg3d':
#         from backbones.vgg3d import VGG3DBackbone
#         return VGG3DBackbone(hidden_dim=hidden_dim)
#     elif backbone_type == 'gaitformer':
#         from backbones.GaitFormer import GaitFormerBackbone
#         return GaitFormerBackbone(hidden_dim=hidden_dim, num_frames=frame_count)
#     elif backbone_type == 'iresnet50':
#         from backbones.resgait import get_iResNet_backbone
#         return get_iResNet_backbone(model_type='iresnet50', hidden_dim=hidden_dim, in_channels=1)
#     elif backbone_type == 'iresnet18':
#         from backbones.resgait import get_iResNet_backbone
#         return get_iResNet_backbone(model_type='iresnet18', hidden_dim=hidden_dim, in_channels=1)
#     elif backbone_type == 'iresnet100':
#         from backbones.resgait import get_iResNet_backbone
#         return get_iResNet_backbone(model_type='iresnet100', hidden_dim=hidden_dim, in_channels=1)
#     else:
#         raise ValueError(f"Unknown backbone_type: {backbone_type}. Supported: {list(backbone_configs.keys())}")

# def train_model(backbone, classifier, dataloaders, backbone_type='gaitstar', num_epochs=50, lr=3e-4, checkpoint_dir='./checkpoints', use_amp=True, resume=False):
#     """
#     Train the model with freeze-unfreeze finetuning strategy.
#     Args:
#         backbone: Backbone for feature extraction.
#         classifier: GaitClassifier with ArcFace loss.
#         dataloaders: Dict with train/val DataLoaders.
#         backbone_type (str): Backbone name for config and checkpoint naming.
#         num_epochs (int): Number of epochs.
#         lr (float): Learning rate.
#         checkpoint_dir (str): Directory to save checkpoints.
#         use_amp (bool): Use mixed precision training.
#         resume (bool): Resume training from checkpoint.
#     Returns:
#         backbone, classifier: Trained models.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Running on {device} with backbone: {backbone_type}, AMP: {use_amp}, Resume: {resume}")
#     backbone = backbone.to(device)
#     classifier = classifier.to(device)
    
#     # Load checkpoint if resuming
#     if resume:
#         checkpoint_path = os.path.join(checkpoint_dir, f'{backbone_type}_best_model.pt')
#         if os.path.exists(checkpoint_path):
#             dict_checkpoint = torch.load(checkpoint_path, map_location=device)
#             backbone.load_state_dict(dict_checkpoint['backbone_state_dict'])
#             classifier.load_state_dict(dict_checkpoint['classifier_state_dict'])
#             print(f"Loaded checkpoint from {checkpoint_path}")
#             del dict_checkpoint
#         else:
#             print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
    
#     # Optimizer and scheduler
#     optimizer = optim.AdamW(
#         list(backbone.parameters()) + list(classifier.parameters()),
#         lr=lr, weight_decay=5e-4
#     )
#     scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
#     scaler = GradScaler() if use_amp else None
    
#     # Finetuning strategy: Freeze CNN layers for first 10 epochs
#     # freeze_epochs = 10 if backbone_type in ['gaitstar', 'gaitset', 'vgg3d', 'iresnet18', 'iresnet50', 'iresnet100'] else 0
#     freeze_epochs = 0
#     best_val_acc = 0.0
#     best_val_loss = np.inf
#     no_improve = 0
#     early_stop_epoch = 5
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     accumulation_steps = 2  # Simulate batch_size=8 with batch_size=4
    
#     for epoch in range(num_epochs):
#         # Set training mode
#         backbone.train()
#         classifier.train()
        
#         # Freeze CNN layers if applicable
#         if epoch < freeze_epochs:
#             for name, param in backbone.named_parameters():
#                 if 'conv' in name.lower():
#                     param.requires_grad = False
#         else:
#             for param in backbone.parameters():
#                 param.requires_grad = True
        
#         # Training
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0
#         optimizer.zero_grad()
#         for i, (frames, labels, _) in enumerate(dataloaders['train']):
#             frames, labels = frames.to(device), labels.to(device)
#             # print(f"Batch {i+1}: Frames shape: {frames.shape}, Min: {frames.min().item()}, Max: {frames.max().item()}")
#             if torch.isnan(frames).any() or torch.isinf(frames).any():
#                 print(f"Warning: NaN or Inf in input frames!")
            
#             if use_amp:
#                 with autocast(device_type=str(device)):
#                     embeddings = backbone(frames)
#                     loss = classifier(embeddings, labels)
#                 scaler.scale(loss / accumulation_steps).backward()
#                 if (i + 1) % accumulation_steps == 0:
#                     torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
#                     torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
#                     scaler.step(optimizer)
#                     scaler.update()
#                     optimizer.zero_grad()
#             else:
#                 embeddings = backbone(frames)
#                 loss = classifier(embeddings, labels)
#                 (loss / accumulation_steps).backward()
#                 if (i + 1) % accumulation_steps == 0:
#                     torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
#                     torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
#                     optimizer.step()
#                     optimizer.zero_grad()
            
#             if torch.isnan(loss) or torch.isinf(loss):
#                 print(f"Warning: NaN or Inf in loss at batch {i+1}")
#             else:
#                 train_loss += loss.item() * frames.size(0) * accumulation_steps
            
#             with torch.no_grad():
#                 if use_amp:
#                     with autocast(device_type=str(device)):
#                         norm_embeddings = nn.functional.normalize(embeddings)
#                         norm_weight = nn.functional.normalize(classifier.weight.to(embeddings.dtype))
#                         logits = nn.functional.linear(norm_embeddings, norm_weight).clamp(-1, 1)
#                         logits = classifier.margin_softmax(logits, labels.view(-1, 1))
#                         _, predicted = torch.max(logits, 1)
#                 else:
#                     norm_embeddings = nn.functional.normalize(embeddings)
#                     norm_weight = nn.functional.normalize(classifier.weight)
#                     logits = nn.functional.linear(norm_embeddings, norm_weight).clamp(-1, 1)
#                     logits = classifier.margin_softmax(logits, labels.view(-1, 1))
#                     _, predicted = torch.max(logits, 1)
#                 train_total += labels.size(0)
#                 train_correct += (predicted == labels).sum().item()
        
#         train_loss = train_loss / train_total if train_total > 0 else float('nan')
#         train_acc = 100 * train_correct / train_total if train_total > 0 else 0.0
        
#         # Validation
#         backbone.eval()
#         classifier.eval()
#         val_loss = 0.0
#         val_correct = 0
#         val_total = 0
#         with torch.no_grad():
#             for frames, labels, _ in dataloaders['val']:
#                 frames, labels = frames.to(device), labels.to(device)
#                 if use_amp:
#                     with autocast(device_type=str(device)):
#                         embeddings = backbone(frames)
#                         loss = classifier(embeddings, labels)
#                 else:
#                     embeddings = backbone(frames)
#                     loss = classifier(embeddings, labels)
                
#                 if not (torch.isnan(loss) or torch.isinf(loss)):
#                     val_loss += loss.item() * frames.size(0)
                
#                 if use_amp:
#                     with autocast(device_type=str(device)):
#                         norm_embeddings = nn.functional.normalize(embeddings)
#                         norm_weight = nn.functional.normalize(classifier.weight.to(embeddings.dtype))
#                         logits = nn.functional.linear(norm_embeddings, norm_weight).clamp(-1, 1)
#                         logits = classifier.margin_softmax(logits, labels.view(-1, 1))
#                         _, predicted = torch.max(logits, 1)
#                 else:
#                     norm_embeddings = nn.functional.normalize(embeddings)
#                     norm_weight = nn.functional.normalize(classifier.weight)
#                     logits = nn.functional.linear(norm_embeddings, norm_weight).clamp(-1, 1)
#                     logits = classifier.margin_softmax(logits, labels.view(-1, 1))
#                     _, predicted = torch.max(logits, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()
        
#         val_loss = val_loss / val_total if val_total > 0 else float('nan')
#         val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
        
#         # Save best model based on validation accuracy
#         if val_loss < best_val_loss and not np.isnan(val_loss):
#             no_improve = 0
#             best_val_acc = val_acc
#             best_val_loss = val_loss
#             checkpoint = {
#                 'backbone_state_dict': backbone.state_dict(),
#                 'classifier_state_dict': classifier.state_dict()
#             }
#             checkpoint_path = os.path.join(checkpoint_dir, f'{backbone_type}_best_model.pt')
#             torch.save(checkpoint, checkpoint_path)
#             print(f"Saved best model to {checkpoint_path}")
#         else:
#             no_improve += 1
        
#         # Adjust learning rate
#         scheduler.step()
        
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
#               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
#         # Early stopping
#         if no_improve == early_stop_epoch:
#             print(f"No improvement on val_acc for {no_improve} epochs. Stopping training!")
#             print(f"Best performance: val_loss: {best_val_loss:.4f}, val_acc: {best_val_acc:.2f}%")
#             break
    
#     return backbone, classifier

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Train gait recognition model")
#     parser.add_argument('--backbone_type', type=str, default='iresnet50', help='Backbone type (gaitstar, gaitset, vgg3d, gaitformer, iresnet18, iresnet50, iresnet100)')
#     parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
#     args = parser.parse_args()

#     # Hyperparameters
#     root_dir = r'.\gallery'
#     batch_size = 8  # Reduced for memory constraints
#     num_epochs = 50
#     lr = 1e-3  # Reduced for stability
#     backbone_type = args.backbone_type
#     use_amp = False  # Toggle AMP
#     resume = args.resume

#     # Get dataloaders
#     dataloaders, label_map = get_dataloaders(root_dir, backbone_type=backbone_type, batch_size=batch_size)
#     num_classes = len(label_map)
#     print(f"Number of identities: {num_classes}")
    
#     # Initialize model
#     config = backbone_configs[backbone_type]
#     backbone = get_backbone(backbone_type, hidden_dim=512, frame_count=config['frame_count'])
#     loss_fn = ArcFace(s=64.0, margin=0.4)  # Reduced s for stability
#     classifier = GaitClassifier(margin_loss=loss_fn, embedding_size=512, num_classes=num_classes)
    
#     # Train
#     backbone, classifier = train_model(
#         backbone, classifier, dataloaders, backbone_type=backbone_type, num_epochs=num_epochs, 
#         lr=lr, use_amp=use_amp, resume=resume
#     )


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
import os
import numpy as np
from dataloader import get_dataloaders, backbone_configs
from classifier import GaitClassifier
from losses import ArcFace

def get_backbone(backbone_type, hidden_dim=512, frame_count=40):
    """
    Factory function to instantiate backbone based on type.
    Args:
        backbone_type (str): Backbone name (e.g., 'gaitstar', 'gaitset', 'iresnet50').
        hidden_dim (int): Embedding dimension.
        frame_count (int): Number of frames.
    Returns:
        nn.Module: Backbone instance.
    """
    if backbone_type == 'gaitstar':
        from backbones.gaitstar import GaitSTARBackbone
        return GaitSTARBackbone(hidden_dim=hidden_dim, frame_count=frame_count)
    elif backbone_type == 'gaitset':
        from backbones.GaitSet_cnn import GaitSetBackbone
        return GaitSetBackbone(hidden_dim=hidden_dim)
    elif backbone_type == 'vgg3d':
        from backbones.vgg3d import VGG3DBackbone
        return VGG3DBackbone(hidden_dim=hidden_dim)
    elif backbone_type == 'gaitformer':
        from backbones.gaitformer import GaitFormerBackbone
        return GaitFormerBackbone(hidden_dim=hidden_dim, num_frames=frame_count)
    elif backbone_type == 'iresnet50':
        from backbones.resgait import get_iResNet_backbone
        return get_iResNet_backbone(model_type='iresnet50', hidden_dim=hidden_dim, in_channels=1)
    elif backbone_type == 'iresnet18':
        from backbones.resgait import get_iResNet_backbone
        return get_iResNet_backbone(model_type='iresnet18', hidden_dim=hidden_dim, in_channels=1)
    elif backbone_type == 'iresnet100':
        from backbones.resgait import get_iResNet_backbone
        return get_iResNet_backbone(model_type='iresnet100', hidden_dim=hidden_dim, in_channels=1)
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}. Supported: {list(backbone_configs.keys())}")

def evaluate_model(backbone, classifier, dataloader, device, use_amp):
    """
    Evaluate the model on the validation set.
    Args:
        backbone: Backbone model.
        classifier: Classifier model.
        dataloader: Validation DataLoader.
        device: Device to run on.
        use_amp: Use mixed precision.
    Returns:
        val_loss: Average validation loss.
        val_acc: Validation accuracy (%).
    """
    backbone.eval()
    classifier.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for frames, labels, _ in dataloader:
            frames, labels = frames.to(device), labels.to(device)
            if use_amp:
                with autocast():
                    embeddings = backbone(frames)
                    loss = classifier(embeddings, labels)
            else:
                embeddings = backbone(frames)
                loss = classifier(embeddings, labels)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                val_loss += loss.item() * frames.size(0)
            
            if use_amp:
                with autocast():
                    norm_embeddings = nn.functional.normalize(embeddings)
                    norm_weight = nn.functional.normalize(classifier.weight.to(embeddings.dtype))
                    logits = nn.functional.linear(norm_embeddings, norm_weight).clamp(-1, 1)
                    logits = classifier.margin_softmax(logits, labels.view(-1, 1))
                    _, predicted = torch.max(logits, 1)
            else:
                norm_embeddings = nn.functional.normalize(embeddings)
                norm_weight = nn.functional.normalize(classifier.weight)
                logits = nn.functional.linear(norm_embeddings, norm_weight).clamp(-1, 1)
                logits = classifier.margin_softmax(logits, labels.view(-1, 1))
                _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / val_total if val_total > 0 else float('nan')
    val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
    return val_loss, val_acc

def train_model(backbone, classifier, dataloaders, backbone_type='gaitstar', num_epochs=50, lr=3e-4, checkpoint_dir='./checkpoints', use_amp=True, resume=False):
    """
    Train the model with freeze-unfreeze finetuning strategy.
    Args:
        backbone: Backbone for feature extraction.
        classifier: GaitClassifier with ArcFace loss.
        dataloaders: Dict with train/val DataLoaders.
        backbone_type (str): Backbone name for config and checkpoint naming.
        num_epochs (int): Number of epochs.
        lr (float): Learning rate.
        checkpoint_dir (str): Directory to save checkpoints.
        use_amp (bool): Use mixed precision training.
        resume (bool): Resume training from checkpoint.
    Returns:
        backbone, classifier: Trained models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} with backbone: {backbone_type}, AMP: {use_amp}, Resume: {resume}")
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=lr, weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = GradScaler() if use_amp else None
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = np.inf
    best_val_acc = 0.0
    no_improve = 0
    
    # Load checkpoint if resuming
    if resume:
        checkpoint_path = os.path.join(checkpoint_dir, f'{backbone_type}_best_model.pt')
        if os.path.exists(checkpoint_path):
            dict_checkpoint = torch.load(checkpoint_path, map_location=device)
            backbone.load_state_dict(dict_checkpoint['backbone_state_dict'])
            classifier.load_state_dict(dict_checkpoint['classifier_state_dict'])
            # optimizer.load_state_dict(dict_checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(dict_checkpoint['scheduler_state_dict'])
            # start_epoch = dict_checkpoint.get('epoch', 0) + 1
            # best_val_loss = dict_checkpoint.get('best_val_loss', np.inf)
            # best_val_acc = dict_checkpoint.get('best_val_acc', 0.0)
            # no_improve = dict_checkpoint.get('no_improve', 0)
            # print(f"Loaded checkpoint from {checkpoint_path} at epoch {dict_checkpoint.get('epoch', 0)}")
            
            # Evaluate checkpoint on validation set
            print("Evaluating checkpoint on validation set...")
            val_loss, val_acc = evaluate_model(backbone, classifier, dataloaders['val'], device, use_amp)
            print(f"Checkpoint validation: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            # Update best_val_loss to the evaluated loss if it's lower
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            del dict_checkpoint
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
            resume = False
    
    early_stop_epoch = 5
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    accumulation_steps = 4  # Simulate batch_size=16 with batch_size=4
    
    for epoch in range(start_epoch, num_epochs):
        # Set training mode
        backbone.train()
        classifier.train()
        
        # Freeze CNN layers for first 10 epochs
        if epoch < 10:
            for name, param in backbone.named_parameters():
                if 'conv' in name.lower():
                    param.requires_grad = False
        else:
            for param in backbone.parameters():
                param.requires_grad = True
        
        # Training
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()
        for i, (frames, labels, _) in enumerate(dataloaders['train']):
            frames, labels = frames.to(device), labels.to(device)
            # print(f"Batch {i+1}: Frames shape: {frames.shape}, Min: {frames.min().item()}, Max: {frames.max().item()}")
            if torch.isnan(frames).any() or torch.isinf(frames).any():
                print(f"Warning: NaN or Inf in input frames!")
            
            if use_amp:
                with autocast():
                    embeddings = backbone(frames)
                    loss = classifier(embeddings, labels)
                scaler.scale(loss / accumulation_steps).backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                embeddings = backbone(frames)
                loss = classifier(embeddings, labels)
                (loss / accumulation_steps).backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf in loss at batch {i+1}")
            else:
                train_loss += loss.item() * frames.size(0) * accumulation_steps
            
            with torch.no_grad():
                if use_amp:
                    with autocast():
                        norm_embeddings = nn.functional.normalize(embeddings)
                        norm_weight = nn.functional.normalize(classifier.weight.to(embeddings.dtype))
                        logits = nn.functional.linear(norm_embeddings, norm_weight).clamp(-1, 1)
                        logits = classifier.margin_softmax(logits, labels.view(-1, 1))
                        _, predicted = torch.max(logits, 1)
                else:
                    norm_embeddings = nn.functional.normalize(embeddings)
                    norm_weight = nn.functional.normalize(classifier.weight)
                    logits = nn.functional.linear(norm_embeddings, norm_weight).clamp(-1, 1)
                    logits = classifier.margin_softmax(logits, labels.view(-1, 1))
                    _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                # print(f"Batch {i+1}: Predicted: {predicted[:5].cpu().numpy()}, Labels: {labels[:5].cpu().numpy()}")
        
        train_loss = train_loss / train_total if train_total > 0 else float('nan')
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0.0
        
        # Validation
        val_loss, val_acc = evaluate_model(backbone, classifier, dataloaders['val'], device, use_amp)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss and not np.isnan(val_loss):
            no_improve = 0
            best_val_loss = val_loss
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                # 'best_val_loss': best_val_loss,
                # 'best_val_acc': best_val_acc,
                # 'no_improve': no_improve
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'{backbone_type}_best_model.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        else:
            no_improve += 1
        
        # Adjust learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        # if no_improve == early_stop_epoch:
        #     print(f"No improvement on val_loss for {no_improve} epochs. Stopping training!")
        #     print(f"Best performance: val_loss: {best_val_loss:.4f}, val_acc: {best_val_acc:.2f}%")
        #     break
    
    return backbone, classifier

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train gait recognition model")
    parser.add_argument('--backbone_type', type=str, default='iresnet50', help='Backbone type (gaitstar, gaitset, vgg3d, gaitformer, iresnet18, iresnet50, iresnet100)')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()

    # Hyperparameters
    root_dir = r'.\gallery'
    batch_size = 8  # Reduced for memory constraints
    num_epochs = 50
    lr = 5e-3  # Reduced for stability
    backbone_type = args.backbone_type
    use_amp = False  # Toggle AMP
    resume = args.resume

    # Get dataloaders
    dataloaders, label_map = get_dataloaders(root_dir, backbone_type=backbone_type, batch_size=batch_size)
    num_classes = len(label_map)
    print(f"Number of identities: {num_classes}")
    
    # Initialize model
    config = backbone_configs[backbone_type]
    backbone = get_backbone(backbone_type, hidden_dim=512, frame_count=config['frame_count'])
    loss_fn = ArcFace(s=12.0, margin=0.05)  # Reduced for stability
    classifier = GaitClassifier(margin_loss=loss_fn, embedding_size=512, num_classes=num_classes)
    
    # Train
    backbone, classifier = train_model(
        backbone, classifier, dataloaders, backbone_type=backbone_type, num_epochs=num_epochs, 
        lr=lr, use_amp=use_amp, resume=resume
    )