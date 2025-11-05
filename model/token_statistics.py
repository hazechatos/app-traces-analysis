"""
Utilities for computing token distribution statistics from sequences.
These statistics can be used as additional features to complement sequential modeling.
"""
import torch
import math
from typing import List, Tuple, Optional


def compute_token_statistics(
    sequences: List[List[int]], 
    vocab_size: int,
    top_k: int = 10,
    include_entropy: bool = True,
    include_diversity: bool = True
) -> torch.Tensor:
    """
    Compute aggregated token distribution statistics for each sequence.
    
    Instead of adding each token's frequency as a separate feature (which would be too many),
    we compute aggregated statistics that capture the distribution:
    - Top-k token frequencies (most frequent tokens)
    - Distribution entropy (measure of randomness/diversity)
    - Total unique tokens
    - Sequence length
    
    Args:
        sequences: List of token sequences (each sequence is a list of token IDs)
        vocab_size: Size of the vocabulary
        top_k: Number of top frequent tokens to include
        include_entropy: Whether to include entropy statistics
        include_diversity: Whether to include diversity metrics
    
    Returns:
        Tensor of shape (n_sequences, n_features) with aggregated statistics
    """
    all_features = []
    
    for seq in sequences:
        if len(seq) == 0:
            # Empty sequence: return zeros
            n_features = top_k + 3  # top_k + entropy + unique_tokens + seq_len
            if include_diversity:
                n_features += 1  # diversity ratio
            features = torch.zeros(n_features)
            all_features.append(features)
            continue
        
        # Convert to torch tensor for efficient counting
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        
        # Filter out padding tokens (0)
        non_padding = seq_tensor[seq_tensor != 0]
        
        if len(non_padding) == 0:
            # Only padding tokens
            n_features = top_k + 3
            if include_diversity:
                n_features += 1
            features = torch.zeros(n_features)
            all_features.append(features)
            continue
        
        # Count token frequencies
        token_counts = torch.bincount(non_padding, minlength=vocab_size + 1)
        token_counts = token_counts[1:]  # Remove padding token (index 0)
        
        # Get top-k most frequent tokens
        top_k_values, top_k_indices = torch.topk(token_counts, k=min(top_k, len(token_counts)))
        top_k_freqs = top_k_values.float() / len(non_padding)  # Normalize to frequencies
        
        # Pad to top_k if needed
        if len(top_k_freqs) < top_k:
            padding = torch.zeros(top_k - len(top_k_freqs))
            top_k_freqs = torch.cat([top_k_freqs, padding])
        
        features = [top_k_freqs]
        
        # Compute entropy (measure of randomness)
        if include_entropy:
            # Normalize to get probabilities
            probs = token_counts.float() / len(non_padding)
            probs = probs[probs > 0]  # Remove zeros for log
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            # Normalize entropy by max possible (log(vocab_size))
            max_entropy = math.log(min(vocab_size, len(non_padding)))
            normalized_entropy = entropy / (max_entropy + 1e-10)
            features.append(torch.tensor([normalized_entropy]))
        
        # Count unique tokens
        unique_tokens = torch.unique(non_padding).shape[0]
        features.append(torch.tensor([unique_tokens / len(non_padding)]))  # Ratio of unique tokens
        
        # Sequence length (normalized)
        features.append(torch.tensor([len(non_padding) / 1000.0]))  # Normalize by typical max length
        
        # Diversity ratio (unique tokens / total tokens)
        if include_diversity:
            diversity = unique_tokens / len(non_padding)
            features.append(torch.tensor([diversity]))
        
        # Concatenate all features
        feature_vector = torch.cat(features)
        all_features.append(feature_vector)
    
    return torch.stack(all_features)


def compute_token_statistics_batch(
    sequences_tensor: torch.Tensor,
    vocab_size: int,
    top_k: int = 10,
    include_entropy: bool = True,
    include_diversity: bool = True
) -> torch.Tensor:
    """
    Compute token distribution statistics for a batch of sequences.
    
    Args:
        sequences_tensor: Tensor of shape (batch_size, seq_len) with token IDs
        vocab_size: Size of the vocabulary
        top_k: Number of top frequent tokens to include
        include_entropy: Whether to include entropy statistics
        include_diversity: Whether to include diversity metrics
    
    Returns:
        Tensor of shape (batch_size, n_features) with aggregated statistics
    """
    batch_size = sequences_tensor.shape[0]
    all_features = []
    
    for i in range(batch_size):
        seq = sequences_tensor[i].cpu().tolist()
        # Filter padding tokens
        seq = [token for token in seq if token != 0]
        
        if len(seq) == 0:
            n_features = top_k + 3
            if include_diversity:
                n_features += 1
            features = torch.zeros(n_features, device=sequences_tensor.device)
            all_features.append(features)
            continue
        
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=sequences_tensor.device)
        
        # Count token frequencies
        token_counts = torch.bincount(seq_tensor, minlength=vocab_size + 1)
        token_counts = token_counts[1:]  # Remove padding token
        
        # Get top-k most frequent tokens
        top_k_values, _ = torch.topk(token_counts, k=min(top_k, len(token_counts)))
        top_k_freqs = top_k_values.float() / len(seq_tensor)
        
        # Pad to top_k if needed
        if len(top_k_freqs) < top_k:
            padding = torch.zeros(top_k - len(top_k_freqs), device=sequences_tensor.device)
            top_k_freqs = torch.cat([top_k_freqs, padding])
        
        features = [top_k_freqs]
        
        # Compute entropy
        if include_entropy:
            probs = token_counts.float() / len(seq_tensor)
            probs = probs[probs > 0]
            if len(probs) > 0:
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                max_entropy = math.log(min(vocab_size, len(seq_tensor)))
                normalized_entropy = entropy / (max_entropy + 1e-10)
            else:
                normalized_entropy = torch.tensor(0.0, device=sequences_tensor.device)
            features.append(normalized_entropy.unsqueeze(0))
        
        # Unique tokens ratio
        unique_tokens = torch.unique(seq_tensor).shape[0]
        features.append(torch.tensor([unique_tokens / len(seq_tensor)], device=sequences_tensor.device))
        
        # Sequence length (normalized)
        features.append(torch.tensor([len(seq_tensor) / 1000.0], device=sequences_tensor.device))
        
        # Diversity ratio
        if include_diversity:
            diversity = unique_tokens / len(seq_tensor)
            features.append(torch.tensor([diversity], device=sequences_tensor.device))
        
        feature_vector = torch.cat(features)
        all_features.append(feature_vector)
    
    return torch.stack(all_features)

