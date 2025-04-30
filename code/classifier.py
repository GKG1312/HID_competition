import math
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.functional import linear, normalize, cross_entropy, softmax

class GaitClassifier(nn.Module):
    def __init__(self, margin_loss, embedding_size, num_classes):
        """
        Classifier for training with ArcFace or linear layer.
        Args:
            margin_loss: callable loss function
            embedding_size: int 
                The dimension of embeddings, required.
            num_classes (int): 
                Number of identities.
        """
        super(GaitClassifier, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).
        Returns:
        -------
        loss: torch.Tensor
            pass
        """
        # print("in classifier",labels.size())
        # labels = labels.squeeze().long()  # Ensure [batch_size]
        # labels = labels.view(-1, 1)  # [batch_size, 1]
        
        # Normalize embeddings and weights
        norm_embeddings = normalize(embeddings)  # [batch_size, embedding_size]
        norm_weight = normalize(self.weight)  # [num_classes, embedding_size]
        
        # Compute logits
        logits = linear(norm_embeddings, norm_weight)  # [batch_size, num_classes]
        logits = logits.clamp(-1, 1)
        
        # Apply margin-based softmax
        logits = self.margin_softmax(logits, labels)
        
        # Compute cross-entropy loss
        # print(labels.size())
        loss = cross_entropy(logits, labels)
        
        return loss