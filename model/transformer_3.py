import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import f1_score
from .token_statistics import compute_token_statistics_batch

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Store as batch-first (1, max_len, d_model) to match module usage
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Grow or move positional encodings if needed (handles larger-than-init seq lens)
        if seq_len > self.pe.size(1) or self.pe.device != x.device:
            device = x.device
            new_len = seq_len
            pe = torch.zeros(new_len, self.d_model, device=device)
            position = torch.arange(0, new_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        return x + self.pe[:, :seq_len, :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # WARNING: Memory bottleneck - attention scores are O(seq_len^2)
        # For seq_len=N, this creates a (batch_size, n_heads, N, N) tensor
        # Memory usage: batch_size * n_heads * N * N * 4 bytes (float32)
        # Example: seq_len=1000 → ~4MB per sequence per head (just for scores!)
        # During backprop, activations + gradients multiply this by ~2-3x
        # Solutions: reduce batch_size, use gradient checkpointing, or implement Flash Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Expand mask to match scores shape: (batch_size, n_heads, seq_len, seq_len)
            # mask comes in as (batch_size, 1, 1, seq_len) - indicates which key positions are valid
            # We need to mask out positions where the key (last dimension) is padding
            # Ensure mask is on the same device as scores
            mask = mask.to(scores.device)
            
            # Get dimensions from scores
            batch_size, n_heads, seq_len_q, seq_len_k = scores.shape
            
            # Validate mask batch size matches
            if mask.size(0) != batch_size:
                raise RuntimeError(f"Mask batch size {mask.size(0)} does not match scores batch size {batch_size}")
            
            if mask.dim() == 4:
                # mask shape: (batch_size, 1, 1, seq_len)
                # Remove middle dimensions: (batch_size, 1, 1, seq_len) -> (batch_size, seq_len)
                mask_1d = mask.squeeze(1).squeeze(1)  # (batch_size, seq_len_original)
                
                # Clamp mask values to 0 or 1 (in case of any floating point issues)
                mask_1d = (mask_1d > 0.5).float()
                
                # Ensure mask_1d matches seq_len_k (trim if too long, pad if too short)
                if mask_1d.size(-1) != seq_len_k:
                    if mask_1d.size(-1) > seq_len_k:
                        mask_1d = mask_1d[:, :seq_len_k]
                    else:
                        # Pad with zeros (padding tokens)
                        padding_size = seq_len_k - mask_1d.size(-1)
                        padding = torch.zeros(batch_size, padding_size, 
                                             device=mask_1d.device, dtype=mask_1d.dtype)
                        mask_1d = torch.cat([mask_1d, padding], dim=-1)
                
                # Create 2D mask: for each query position, mask all padding key positions
                # mask_1d: (batch_size, seq_len_k) 
                # We want: (batch_size, n_heads, seq_len_q, seq_len_k)
                # where each row (query position) has the same mask values
                mask_2d = mask_1d.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len_k)
                # Use repeat instead of expand to ensure contiguous tensor
                mask_expanded = mask_2d.repeat(1, n_heads, seq_len_q, 1)  # (batch_size, n_heads, seq_len_q, seq_len_k)
            else:
                # Fallback: try to handle other shapes
                if mask.dim() == 4:
                    mask_expanded = mask.repeat(1, n_heads, 1, 1)
                else:
                    raise ValueError(f"Unexpected mask shape: {mask.shape}, expected 4D tensor")
            
            # Ensure shapes match exactly
            if mask_expanded.shape != scores.shape:
                raise RuntimeError(f"Mask shape {mask_expanded.shape} does not match scores shape {scores.shape}")
            
            # Convert mask to boolean: 1 -> True (keep), 0 -> False (mask out)
            # Ensure the mask is contiguous and properly typed
            mask_bool = (mask_expanded > 0.5).contiguous()
            
            # Apply mask: mask out positions where mask_bool is False (padding positions)
            scores = scores.masked_fill(~mask_bool, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        return output

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward layers."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class UsernameTransformer(nn.Module):
    """
    Transformer model for username prediction from action sequences with multitoken context.
    
    During training: Takes (username, action_sequence, browser, duration_bucket, speed_bucket) tuples
    During inference: Takes (action_sequence, browser, duration_bucket, speed_bucket) and predicts username
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1000,
        dropout: float = 0.1,
        n_usernames: int = None,
        discrete_contexts: dict = None, # multitoken context for (browser, duration, action_speed)
        use_token_statistics: bool = True, # Whether to include token distribution statistics
        token_stats_top_k: int = 10, # Number of top frequent tokens to include
        token_stats_dim: int = None, # Dimension of token statistics features (auto-computed if None)
        context_as_features: bool = True # If True, context tokens are concatenated as features (like statistics). If False, they're added as sequence tokens.
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_usernames = n_usernames
        self.discrete_contexts = discrete_contexts if discrete_contexts is not None else {}
        self.n_contexts = len(self.discrete_contexts)
        self.use_token_statistics = use_token_statistics
        self.token_stats_top_k = token_stats_top_k
        self.context_as_features = context_as_features
        
        # Compute token statistics feature dimension
        # top_k + entropy + unique_ratio + seq_len + diversity
        if token_stats_dim is None:
            self.token_stats_dim = token_stats_top_k + 4  # top_k + entropy + unique_ratio + seq_len + diversity
        else:
            self.token_stats_dim = token_stats_dim

        # Token embedding for action sequences
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Context embeddings
        self.context_embeddings = nn.ModuleDict({
            name: nn.Embedding(n_values, d_model)
            for name, n_values in self.discrete_contexts.items()
        })
        
        # Username embedding (for training)
        self.username_embedding = nn.Embedding(n_usernames, d_model)
        
        # Positional encoding (only increased if context tokens are in sequence)
        if self.context_as_features:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len + self.n_contexts)
        
        # If context as features, create embedding layer to map context embeddings to feature space
        if self.context_as_features and self.n_contexts > 0:
            # Each context is embedded to d_model, we'll concatenate all and embed to d_model // 4
            self.context_features_embedding = nn.Sequential(
                nn.Linear(self.n_contexts * d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4)
            )
            context_features_size = d_model // 4
        else:
            context_features_size = 0
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Token statistics embedding (if enabled)
        if self.use_token_statistics:
            # Embed token statistics into d_model space
            self.token_stats_embedding = nn.Sequential(
                nn.Linear(self.token_stats_dim, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4)
            )
            stats_features_size = d_model // 4
        else:
            stats_features_size = 0
        
        # Combined input size for classifier: d_model (from transformer) + stats + context features
        classifier_input_size = d_model + stats_features_size + context_features_size
        
        # Classification head for username prediction (MLP)
        if n_usernames is not None:
            self.username_classifier = nn.Sequential(
                nn.Linear(classifier_input_size, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_usernames)
            )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq):
        """Create padding mask for variable length sequences."""
        return (seq != 0).unsqueeze(1).unsqueeze(2)
    
    def forward(self, action_sequence, browser=None, duration_bucket=None, speed_bucket=None, username=None, training=True):
        """
        Forward pass of the transformer with multitoken context.
        
        Args:
            action_sequence: Tensor of shape (batch_size, seq_len) with action token IDs
            browser: Tensor of shape (batch_size,) with browser IDs
            duration_bucket: Tensor of shape (batch_size,) with duration bucket IDs
            speed_bucket: Tensor of shape (batch_size,) with speed bucket IDs
            username: Tensor of shape (batch_size,) with username IDs (only used during training)
            training: Whether in training mode
        
        Returns:
            If training: (username_logits, action_embeddings)
            If inference: username_logits
        """
        batch_size, seq_len = action_sequence.shape
        
        # Collect context embeddings
        context_features_list = []
        context_embs = []
        context_masks = []
        
        if browser is not None:
            browser_emb = self.context_embeddings['browser'](browser) * math.sqrt(self.d_model)  # (batch_size, d_model)
            context_features_list.append(browser_emb)
            if not self.context_as_features:
                browser_emb = browser_emb.unsqueeze(1)  # (batch_size, 1, d_model)
                context_embs.append(browser_emb)
                context_masks.append(torch.ones(batch_size, 1, 1, 1, device=action_sequence.device))
        
        if duration_bucket is not None:
            duration_emb = self.context_embeddings['duration_bucket'](duration_bucket) * math.sqrt(self.d_model)  # (batch_size, d_model)
            context_features_list.append(duration_emb)
            if not self.context_as_features:
                duration_emb = duration_emb.unsqueeze(1)  # (batch_size, 1, d_model)
                context_embs.append(duration_emb)
                context_masks.append(torch.ones(batch_size, 1, 1, 1, device=action_sequence.device))
        
        if speed_bucket is not None:
            speed_emb = self.context_embeddings['speed_bucket'](speed_bucket) * math.sqrt(self.d_model)  # (batch_size, d_model)
            context_features_list.append(speed_emb)
            if not self.context_as_features:
                speed_emb = speed_emb.unsqueeze(1)  # (batch_size, 1, d_model)
                context_embs.append(speed_emb)
                context_masks.append(torch.ones(batch_size, 1, 1, 1, device=action_sequence.device))

        # Token embeddings for actions
        action_emb = self.token_embedding(action_sequence) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        
        # Process through transformer
        if self.context_as_features:
            # Context tokens are NOT in the sequence - only action tokens
            x = action_emb
            mask = self.create_padding_mask(action_sequence)  # (batch_size, 1, 1, seq_len)
        else:
            # Context tokens ARE in the sequence (original approach)
            if context_embs:
                x = torch.cat(context_embs + [action_emb], dim=1)  # (batch_size, n_contexts + seq_len, d_model)
            else:
                x = action_emb
            
            # Create padding mask (extend for context tokens)
            action_mask = self.create_padding_mask(action_sequence)  # (batch_size, 1, 1, seq_len)
            if context_masks:
                mask = torch.cat(context_masks + [action_mask], dim=-1)  # (batch_size, 1, 1, n_contexts + seq_len)
            else:
                mask = action_mask
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Apply layer norm
        x = self.layer_norm(x)
        
        # Global average pooling over sequence length
        mask_expanded = mask.squeeze(1).squeeze(1).float()  # (batch_size, seq_len) or (batch_size, n_contexts + seq_len)
        
        if self.context_as_features:
            # Pool over all action tokens (no context tokens in sequence)
            action_tokens = x  # (batch_size, seq_len, d_model)
            action_mask = mask_expanded  # (batch_size, seq_len)
        else:
            # Pool only over action tokens (skip context tokens at the beginning)
            n_actual_contexts = len(context_embs)
            action_tokens = x[:, n_actual_contexts:, :]  # (batch_size, seq_len, d_model)
            action_mask = mask_expanded[:, n_actual_contexts:]  # (batch_size, seq_len)
        
        x_masked = action_tokens * action_mask.unsqueeze(-1)
        denom = action_mask.sum(dim=1, keepdim=True)
        zero_mask = (denom == 0)
        denom = denom.clamp(min=1.0)
        pooled = x_masked.sum(dim=1) / denom
        pooled = pooled.masked_fill(zero_mask, 0.0)
        
        # Collect all features to concatenate
        feature_list = [pooled]
        
        # Add context as features (if enabled)
        if self.context_as_features and len(context_features_list) > 0:
            # Concatenate all context embeddings
            context_features = torch.cat(context_features_list, dim=1)  # (batch_size, n_contexts * d_model)
            # Embed context features
            context_embedded = self.context_features_embedding(context_features)  # (batch_size, d_model // 4)
            feature_list.append(context_embedded)
        
        # Add token distribution statistics if enabled
        if self.use_token_statistics:
            # Compute token statistics for the batch
            token_stats = compute_token_statistics_batch(
                action_sequence,
                vocab_size=self.vocab_size,
                top_k=self.token_stats_top_k,
                include_entropy=True,
                include_diversity=True
            ).to(pooled.device)
            
            # Embed statistics into d_model space
            stats_embedded = self.token_stats_embedding(token_stats)  # (batch_size, d_model // 4)
            feature_list.append(stats_embedded)
        
        # Concatenate all features
        combined_features = torch.cat(feature_list, dim=1)  # (batch_size, d_model + context_features + stats_features)
        
        if training and username is not None:
            # Training mode: return both username logits and embeddings
            username_logits = self.username_classifier(combined_features)
            return username_logits, pooled
        else:
            # Inference mode: return only username logits
            username_logits = self.username_classifier(combined_features)
            return username_logits
    
    def predict_username(self, action_sequence, browser=None, duration_bucket=None, speed_bucket=None):
        """
        Predict username from action sequence and context tokens (inference mode).
        
        Args:
            action_sequence: Tensor of shape (batch_size, seq_len) or (seq_len,)
            browser: Tensor of shape (batch_size,) or scalar with browser IDs
            duration_bucket: Tensor of shape (batch_size,) or scalar with duration bucket IDs
            speed_bucket: Tensor of shape (batch_size,) or scalar with speed bucket IDs
        
        Returns:
            Predicted username logits and probabilities
        """
        self.eval()
        
        if action_sequence.dim() == 1:
            action_sequence = action_sequence.unsqueeze(0)
        
        if browser is not None and browser.dim() == 0:
            browser = browser.unsqueeze(0)
        
        if duration_bucket is not None and duration_bucket.dim() == 0:
            duration_bucket = duration_bucket.unsqueeze(0)
        
        if speed_bucket is not None and speed_bucket.dim() == 0:
            speed_bucket = speed_bucket.unsqueeze(0)
        
        with torch.no_grad():
            username_logits = self.forward(action_sequence, browser, duration_bucket, speed_bucket, training=False)
            username_probs = F.softmax(username_logits, dim=-1)
            
        return username_logits, username_probs

class UsernameTransformerTrainer:
    """Training utilities for the UsernameTransformer."""
    
    def __init__(self, model, learning_rate=1e-5, device='cpu', class_weights=None):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Use class weights if provided, otherwise use standard CrossEntropyLoss
        if class_weights is not None:
            class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, action_sequences: torch.Tensor, usernames: torch.Tensor, browsers: torch.Tensor = None, duration_buckets: torch.Tensor = None, speed_buckets: torch.Tensor = None):
        """Single training step."""
        self.model.train()
        
        action_sequences = action_sequences.to(self.device)
        usernames = usernames.to(self.device)
        browsers = browsers.to(self.device) if browsers is not None else None
        duration_buckets = duration_buckets.to(self.device) if duration_buckets is not None else None
        speed_buckets = speed_buckets.to(self.device) if speed_buckets is not None else None
        
        self.optimizer.zero_grad()
        
        username_logits, _ = self.model(action_sequences, browsers, duration_buckets, speed_buckets, usernames, training=True)
        loss = self.criterion(username_logits, usernames)
        
        loss.backward()
        # Check gradients before optimizer step
        # check_gradient_flow(self.model)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, action_sequences, usernames, browsers=None, duration_buckets=None, speed_buckets=None):
        """Evaluate model on validation data."""
        self.model.eval()
        
        action_sequences = action_sequences.to(self.device)
        usernames = usernames.to(self.device)
        browsers = browsers.to(self.device) if browsers is not None else None
        duration_buckets = duration_buckets.to(self.device) if duration_buckets is not None else None
        speed_buckets = speed_buckets.to(self.device) if speed_buckets is not None else None
        
        with torch.no_grad():
            username_logits = self.model(action_sequences, browsers, duration_buckets, speed_buckets, training=False)
            loss = self.criterion(username_logits, usernames)
            
            predictions = torch.argmax(username_logits, dim=-1)
            accuracy = (predictions == usernames).float().mean()
            
            # Calculate macro-F1 score
            predictions_cpu = predictions.cpu().numpy()
            usernames_cpu = usernames.cpu().numpy()
            macro_f1 = f1_score(usernames_cpu, predictions_cpu, average='macro', zero_division=0)
        
        return loss.item(), accuracy.item(), macro_f1, predictions

# Helper function to calculate class weights
def calculate_class_weights(username_tokens):
    """
    Calculate class weights for imbalanced username data using sklearn-style balanced weights.
    
    Args:
        username_tokens: List of username token IDs
        
    Returns:
        class_weights: Tensor of weights for each class
    """
    username_counts = torch.bincount(torch.tensor(username_tokens, dtype=torch.long))
    total_samples = len(username_tokens)
    num_classes = len(username_counts)
    
    # sklearn balanced weights: n_samples / (n_classes * np.bincount(y))
    # No normalization - let PyTorch handle the scaling, stronger weights help with severe imbalance
    class_weights = total_samples / (num_classes * username_counts.float())
    
    return class_weights

# Example usage and training function
def create_model(vocab_size, n_usernames, discrete_contexts=None, **kwargs):
    """Create a UsernameTransformer model with default parameters."""
    return UsernameTransformer(
        vocab_size=vocab_size,
        n_usernames=n_usernames,
        discrete_contexts=discrete_contexts,
        d_model=kwargs.get('d_model', 64),
        n_heads=kwargs.get('n_heads', 2),
        n_layers=kwargs.get('n_layers', 2),
        d_ff=kwargs.get('d_ff', 512),
        max_seq_len=kwargs.get('max_seq_len', 500),
        dropout=kwargs.get('dropout', 0.15),
        use_token_statistics=kwargs.get('use_token_statistics', True),
        token_stats_top_k=kwargs.get('token_stats_top_k', 10),
        token_stats_dim=kwargs.get('token_stats_dim', None),
        context_as_features=kwargs.get('context_as_features', True)  # Default: context as features for consistency
    )

def train_model(model, train_data, val_data, learning_rate=1e-5, epochs=50, batch_size=8, max_seq_len=100, device='cpu'):
    """
    Train the UsernameTransformer model with multitoken context.
    
    Args:
        model: UsernameTransformer model
        train_data: (action_sequences, usernames, browsers, duration_buckets, speed_buckets) tuple or variations
        val_data: (action_sequences, usernames, browsers, duration_buckets, speed_buckets) tuple or variations
        epochs: Number of training epochs
        batch_size: Batch size for training (reduced to prevent memory issues)
        max_seq_len: Maximum sequence length to process
            WARNING: Memory usage scales QUADRATICALLY with max_seq_len due to attention mechanism.
            Each doubling of max_seq_len requires 4x more GPU memory. If you hit OOM errors:
            - Reduce batch_size proportionally (e.g., if doubling seq_len, halve batch_size)
            - Use gradient checkpointing (torch.utils.checkpoint)
            - Consider implementing Flash Attention for linear memory scaling
        device: Device to train on
    """
    username_tokens = train_data[1]
    weights = calculate_class_weights(username_tokens)
    trainer = UsernameTransformerTrainer(model, learning_rate=learning_rate, device=device, class_weights=weights)
    
    # Handle different input formats
    if len(train_data) == 3:
        train_sequences, train_usernames, train_browsers = train_data
        train_duration_buckets = None
        train_speed_buckets = None
    elif len(train_data) == 5:
        train_sequences, train_usernames, train_browsers, train_duration_buckets, train_speed_buckets = train_data
    else:
        raise ValueError(f"train_data must have 3 or 5 elements, got {len(train_data)}")
    
    if len(val_data) == 3:
        val_sequences, val_usernames, val_browsers = val_data
        val_duration_buckets = None
        val_speed_buckets = None
    elif len(val_data) == 5:
        val_sequences, val_usernames, val_browsers, val_duration_buckets, val_speed_buckets = val_data
    else:
        raise ValueError(f"val_data must have 3 or 5 elements, got {len(val_data)}")
    
    print(f"Training on {len(train_sequences)} samples")
    print(f"Validation on {len(val_sequences)} samples")
    print(f"Using max sequence length: {max_seq_len}")
    print(f"Using batch size: {batch_size}")
    
    # Function to truncate sequences to max_seq_len
    def truncate_sequences(sequences, max_len):
        truncated = []
        for seq in sequences:
            if len(seq) > max_len:
                # Take the last max_len tokens (most recent actions)
                truncated.append(seq[-max_len:])
            else:
                truncated.append(seq)
        return truncated
    
    # Truncate sequences to prevent memory issues
    train_sequences = truncate_sequences(train_sequences, max_seq_len)
    val_sequences = truncate_sequences(val_sequences, max_seq_len)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_sequences), batch_size):
            batch_sequences = train_sequences[i:i+batch_size]
            batch_usernames = train_usernames[i:i+batch_size]
            batch_browsers = train_browsers[i:i+batch_size] if train_browsers is not None else None
            batch_duration_buckets = train_duration_buckets[i:i+batch_size] if train_duration_buckets is not None else None
            batch_speed_buckets = train_speed_buckets[i:i+batch_size] if train_speed_buckets is not None else None
            
            # Convert to tensors and pad sequences
            batch_sequences = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(seq, dtype=torch.long) for seq in batch_sequences],
                batch_first=True,
                padding_value=0
            )
            
            batch_usernames = torch.tensor(batch_usernames, dtype=torch.long)
            batch_browsers = torch.tensor(batch_browsers, dtype=torch.long) if batch_browsers is not None else None
            batch_duration_buckets = torch.tensor(batch_duration_buckets, dtype=torch.long) if batch_duration_buckets is not None else None
            batch_speed_buckets = torch.tensor(batch_speed_buckets, dtype=torch.long) if batch_speed_buckets is not None else None
            
            loss = trainer.train_step(batch_sequences, batch_usernames, batch_browsers, batch_duration_buckets, batch_speed_buckets)
            train_loss += loss
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # Validation - process in batches to avoid memory issues
        val_loss = 0
        val_accuracy = 0
        num_val_batches = 0
        all_predictions = []
        all_true_labels = []
        
        for i in range(0, len(val_sequences), batch_size):
            val_batch_sequences = val_sequences[i:i+batch_size]
            val_batch_usernames = val_usernames[i:i+batch_size]
            val_batch_browsers = val_browsers[i:i+batch_size] if val_browsers is not None else None
            val_batch_duration_buckets = val_duration_buckets[i:i+batch_size] if val_duration_buckets is not None else None
            val_batch_speed_buckets = val_speed_buckets[i:i+batch_size] if val_speed_buckets is not None else None
            
            # Convert to tensors and pad sequences
            val_batch_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(seq, dtype=torch.long) for seq in val_batch_sequences],
                batch_first=True,
                padding_value=0
            )
            
            val_batch_usernames = torch.tensor(val_batch_usernames, dtype=torch.long)
            val_batch_browsers = torch.tensor(val_batch_browsers, dtype=torch.long) if val_batch_browsers is not None else None
            val_batch_duration_buckets = torch.tensor(val_batch_duration_buckets, dtype=torch.long) if val_batch_duration_buckets is not None else None
            val_batch_speed_buckets = torch.tensor(val_batch_speed_buckets, dtype=torch.long) if val_batch_speed_buckets is not None else None
            
            batch_loss, batch_accuracy, batch_macro_f1, batch_predictions = trainer.evaluate(val_batch_tensor, val_batch_usernames, val_batch_browsers, val_batch_duration_buckets, val_batch_speed_buckets)
            val_loss += batch_loss
            val_accuracy += batch_accuracy
            all_predictions.append(batch_predictions.cpu())
            all_true_labels.append(val_batch_usernames.cpu())
            num_val_batches += 1
        
        val_loss = val_loss / num_val_batches
        val_accuracy = val_accuracy / num_val_batches
        
        # Calculate overall macro-F1 on full validation set
        all_predictions = torch.cat(all_predictions, dim=0)
        all_true_labels = torch.cat(all_true_labels, dim=0)
        val_macro_f1 = f1_score(all_true_labels.numpy(), all_predictions.numpy(), average='macro', zero_division=0)
        # Determine minimum length for bincount
        minlength = model.n_usernames if model.n_usernames is not None else (all_predictions.max().item() + 1 if len(all_predictions) > 0 else 1)
        prediction_counts = torch.bincount(all_predictions, minlength=minlength)
        total_predictions = len(all_predictions)
        prediction_proportions = prediction_counts.float() / total_predictions
        
        # Get top predicted usernames (only consider usernames that were actually predicted)
        non_zero_mask = prediction_counts > 0
        if non_zero_mask.any():
            top_k = min(5, non_zero_mask.sum().item())
            top_proportions, top_indices = torch.topk(prediction_proportions, k=top_k)
        else:
            top_proportions = torch.tensor([])
            top_indices = torch.tensor([], dtype=torch.long)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Macro-F1: {val_macro_f1:.4f}")
            if len(top_indices) > 0:
                print(f"  Top predicted usernames (by frequency):")
                for idx, (username_idx, proportion) in enumerate(zip(top_indices.tolist(), top_proportions.tolist())):
                    print(f"    {idx+1}. Username {username_idx}: {proportion:.2%} ({prediction_counts[username_idx].item()}/{total_predictions})")
    
    return model




def check_gradient_flow(model):
    """Check if gradients are flowing through the model."""
    print("\n=== Gradient Flow Check ===")
    total_norm = 0
    param_norm = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if gradient exists
            if param.grad is None:
                print(f"⚠️  {name}: NO GRADIENT (None)")
                zero_grad_count += 1
                continue
            
            # Check gradient norm
            param_norm = param.data.norm().item()
            grad_norm = param.grad.norm().item()
            
            if grad_norm == 0:
                print(f"⚠️  {name}: ZERO gradient (norm=0)")
                zero_grad_count += 1
            elif grad_norm < 1e-7:
                print(f"⚠️  {name}: VANISHING gradient (norm={grad_norm:.2e})")
            elif grad_norm > 100:
                print(f"⚠️  {name}: EXPLODING gradient (norm={grad_norm:.2e})")
            else:
                print(f"✓  {name}: OK (grad_norm={grad_norm:.4f}, param_norm={param_norm:.4f})")
            
            # Ratio of gradient to parameter
            ratio = grad_norm / (param_norm + 1e-10)
            if ratio < 1e-6:
                print(f"   ⚠️  Very small gradient/param ratio: {ratio:.2e}")
            
            total_norm += grad_norm ** 2
    
    total_norm = total_norm ** 0.5
    print(f"\nTotal gradient norm: {total_norm:.4f}")
    print(f"Parameters with zero/None gradients: {zero_grad_count}")
    
    return total_norm, zero_grad_count