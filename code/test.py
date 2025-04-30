import torch
import os
import numpy as np
from dataloader import get_dataloaders
from backbones.GaitSet_cnn import GaitSetBackbone
from classifier import GaitClassifier
from losses import ArcFace

def test_model(backbone, classifier, test_loader, checkpoint_path, output_dir='./outputs'):
    """
    Test the model using cosine similarity for verification.
    Args:
        backbone: GaitSetBackbone for feature extraction.
        classifier: GaitClassifier with ArcFace loss.
        test_loader: DataLoader for test set.
        checkpoint_path (str): Path to trained model weights.
        output_dir (str): Directory to save predictions.
    Returns:
        float: Verification accuracy at threshold 0.8.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    backbone.eval()
    classifier.eval()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    embeddings = []
    labels = []
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for frames, batch_labels in test_loader:
            frames, batch_labels = frames.to(device), batch_labels.to(device)
            # Debug labels
            print(f"Test batch labels: {batch_labels.tolist()}, shape: {batch_labels.shape}")
            batch_embeddings = backbone(frames)  # [B, hidden_dim]
            embeddings.append(batch_embeddings.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)  # [N, hidden_dim]
    labels = np.concatenate(labels, axis=0)  # [N]
    
    # Compute pairwise cosine similarities
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cosine_sim = np.dot(norm_embeddings, norm_embeddings.T)  # [N, N]
    
    # Verification accuracy at threshold 0.8
    threshold = 0.8
    correct = 0
    total = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            same_identity = labels[i] == labels[j]
            sim = cosine_sim[i, j]
            if (sim > threshold and same_identity) or (sim <= threshold and not same_identity):
                correct += 1
            total += 1
    
    accuracy = 100 * correct / total
    print(f"Verification Accuracy (threshold={threshold}): {accuracy:.2f}%")
    
    # Save embeddings and labels
    np.save(os.path.join(output_dir, 'test_embeddings.npy'), embeddings)
    np.save(os.path.join(output_dir, 'test_labels.npy'), labels)
    
    return accuracy

if __name__ == "__main__":
    # Hyperparameters
    root_dir = r'.\gallery'
    batch_size = 8
    frame_count = 30
    checkpoint_path = './checkpoints/best_model.pt'
    
    # Get test dataloader
    dataloaders, label_map = get_dataloaders(root_dir, batch_size, frame_count)
    num_classes = len(label_map)  # Total identities (859)
    print(f"Number of identities: {num_classes}")
    
    # Initialize model
    loss_fn = ArcFace(s=64.0, margin=0.4)
    backbone = GaitSetBackbone(hidden_dim=512)
    classifier = GaitClassifier(margin_loss=loss_fn, embedding_size=512, num_classes=num_classes)
    
    # Test
    accuracy = test_model(backbone, classifier, test_loader=dataloaders['test'], checkpoint_path=checkpoint_path)