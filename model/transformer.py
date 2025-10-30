import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
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
    Transformer model for username prediction from action sequences with browser context.
    
    During training: Takes (username, action_sequence, browser) tuples
    During inference: Takes (action_sequence, browser) and predicts username
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
        n_browsers: int = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_usernames = n_usernames
        self.n_browsers = n_browsers
        
        # Token embedding for action sequences
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Browser embedding (for context)
        if n_browsers is not None:
            self.browser_embedding = nn.Embedding(n_browsers, d_model)
        
        # Username embedding (for training)
        if n_usernames is not None:
            self.username_embedding = nn.Embedding(n_usernames, d_model)
        
        # Positional encoding (increased by 1 to account for browser token)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len + 1)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Classification head for username prediction
        if n_usernames is not None:
            self.username_classifier = nn.Linear(d_model, n_usernames)
        
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
    
    def forward(self, action_sequence, browser=None, username=None, training=True):
        """
        Forward pass of the transformer with browser context.
        
        Args:
            action_sequence: Tensor of shape (batch_size, seq_len) with action token IDs
            browser: Tensor of shape (batch_size,) with browser IDs
            username: Tensor of shape (batch_size,) with username IDs (only used during training)
            training: Whether in training mode
        
        Returns:
            If training: (username_logits, action_embeddings)
            If inference: username_logits
        """
        batch_size, seq_len = action_sequence.shape
        
        # Create browser token embeddings
        if browser is not None and self.n_browsers is not None:
            browser_emb = self.browser_embedding(browser) * math.sqrt(self.d_model)  # (batch_size, d_model)
            browser_emb = browser_emb.unsqueeze(1)  # (batch_size, 1, d_model)
        else:
            # If no browser provided, create zero embeddings
            browser_emb = torch.zeros(batch_size, 1, self.d_model, device=action_sequence.device)
        
        # Token embeddings for actions
        action_emb = self.token_embedding(action_sequence) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        
        # Concatenate browser token at the beginning
        x = torch.cat([browser_emb, action_emb], dim=1)  # (batch_size, seq_len + 1, d_model)
        
        # Create padding mask (extend for browser token)
        action_mask = self.create_padding_mask(action_sequence)  # (batch_size, 1, 1, seq_len)
        browser_mask = torch.ones(batch_size, 1, 1, 1, device=action_sequence.device)  # Browser token is never masked
        mask = torch.cat([browser_mask, action_mask], dim=-1)  # (batch_size, 1, 1, seq_len + 1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Apply layer norm
        x = self.layer_norm(x)
        
        # Global average pooling over sequence length (excluding browser token)
        # Use mask to ignore padding tokens, but exclude browser token from pooling
        mask_expanded = mask.squeeze(1).squeeze(1).float()  # (batch_size, seq_len + 1)
        # Only pool over action tokens (skip browser token at position 0)
        action_tokens = x[:, 1:, :]  # (batch_size, seq_len, d_model)
        action_mask = mask_expanded[:, 1:]  # (batch_size, seq_len)
        x_masked = action_tokens * action_mask.unsqueeze(-1)
        denom = action_mask.sum(dim=1, keepdim=True)
        zero_mask = (denom == 0)
        denom = denom.clamp(min=1.0)
        pooled = x_masked.sum(dim=1) / denom
        pooled = pooled.masked_fill(zero_mask, 0.0)
        
        if training and username is not None:
            # Training mode: return both username logits and embeddings
            username_logits = self.username_classifier(pooled)
            return username_logits, pooled
        else:
            # Inference mode: return only username logits
            username_logits = self.username_classifier(pooled)
            return username_logits
    
    def predict_username(self, action_sequence, browser=None):
        """
        Predict username from action sequence and browser context (inference mode).
        
        Args:
            action_sequence: Tensor of shape (batch_size, seq_len) or (seq_len,)
            browser: Tensor of shape (batch_size,) or scalar with browser IDs
        
        Returns:
            Predicted username logits and probabilities
        """
        self.eval()
        
        if action_sequence.dim() == 1:
            action_sequence = action_sequence.unsqueeze(0)
        
        if browser is not None and browser.dim() == 0:
            browser = browser.unsqueeze(0)
        
        with torch.no_grad():
            username_logits = self.forward(action_sequence, browser, training=False)
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
    
    def train_step(self, action_sequences: torch.Tensor, usernames: torch.Tensor, browsers: torch.Tensor = None):
        """Single training step."""
        self.model.train()
        
        action_sequences = action_sequences.to(self.device)
        usernames = usernames.to(self.device)
        if browsers is not None:
            browsers = browsers.to(self.device)
        
        self.optimizer.zero_grad()
        
        username_logits, _ = self.model(action_sequences, browsers, usernames, training=True)
        loss = self.criterion(username_logits, usernames)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, action_sequences, usernames, browsers=None):
        """Evaluate model on validation data."""
        self.model.eval()
        
        action_sequences = action_sequences.to(self.device)
        usernames = usernames.to(self.device)
        if browsers is not None:
            browsers = browsers.to(self.device)
        
        with torch.no_grad():
            username_logits = self.model(action_sequences, browsers, training=False)
            loss = self.criterion(username_logits, usernames)
            
            predictions = torch.argmax(username_logits, dim=-1)
            accuracy = (predictions == usernames).float().mean()
        
        return loss.item(), accuracy.item()

# Helper function to calculate class weights
def calculate_class_weights(username_tokens):
    """
    Calculate class weights for imbalanced username data.
    
    Args:
        username_tokens: List of username token IDs
        
    Returns:
        class_weights: Tensor of weights for each class
    """
    username_counts = torch.bincount(torch.tensor(username_tokens, dtype=torch.long))
    total_samples = len(username_tokens)
    num_classes = len(username_counts)
    
    # Calculate inverse frequency weights
    class_weights = total_samples / (num_classes * username_counts.float())
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return class_weights

# Example usage and training function
def create_model(vocab_size, n_usernames, n_browsers=None, **kwargs):
    """Create a UsernameTransformer model with default parameters."""
    return UsernameTransformer(
        vocab_size=vocab_size,
        n_usernames=n_usernames,
        n_browsers=n_browsers,
        d_model=kwargs.get('d_model', 64),
        n_heads=kwargs.get('n_heads', 2),
        n_layers=kwargs.get('n_layers', 2),
        d_ff=kwargs.get('d_ff', 512),
        max_seq_len=kwargs.get('max_seq_len', 500),
        dropout=kwargs.get('dropout', 0.15)
    )

def train_model(model, train_data, val_data, learning_rate=1e-5, epochs=50, batch_size=8, max_seq_len=100, device='cpu'):
    """
    Train the UsernameTransformer model with browser context.
    
    Args:
        model: UsernameTransformer model
        train_data: (action_sequences, usernames, browsers) tuple or (action_sequences, usernames) tuple
        val_data: (action_sequences, usernames, browsers) tuple or (action_sequences, usernames) tuple  
        epochs: Number of training epochs
        batch_size: Batch size for training (reduced to prevent memory issues)
        max_seq_len: Maximum sequence length to process
        device: Device to train on
    """
    username_tokens = train_data[1]
    weights = calculate_class_weights(username_tokens)
    trainer = UsernameTransformerTrainer(model, learning_rate=learning_rate, device=device, class_weights=weights)
    
    # Handle both old format (2 elements) and new format (3 elements)
    if len(train_data) == 3:
        train_sequences, train_usernames, train_browsers = train_data
    else:
        train_sequences, train_usernames = train_data
        train_browsers = None
        
    if len(val_data) == 3:
        val_sequences, val_usernames, val_browsers = val_data
    else:
        val_sequences, val_usernames = val_data
        val_browsers = None
    
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
            
            # Convert to tensors and pad sequences
            batch_sequences = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(seq, dtype=torch.long) for seq in batch_sequences],
                batch_first=True,
                padding_value=0
            )
            
            batch_usernames = torch.tensor(batch_usernames, dtype=torch.long)
            
            if batch_browsers is not None:
                batch_browsers = torch.tensor(batch_browsers, dtype=torch.long)
            
            loss = trainer.train_step(batch_sequences, batch_usernames, batch_browsers)
            train_loss += loss
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # Validation - process in batches to avoid memory issues
        val_loss = 0
        val_accuracy = 0
        num_val_batches = 0
        
        for i in range(0, len(val_sequences), batch_size):
            val_batch_sequences = val_sequences[i:i+batch_size]
            val_batch_usernames = val_usernames[i:i+batch_size]
            val_batch_browsers = val_browsers[i:i+batch_size] if val_browsers is not None else None
            
            # Convert to tensors and pad sequences
            val_batch_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(seq, dtype=torch.long) for seq in val_batch_sequences],
                batch_first=True,
                padding_value=0
            )
            
            val_batch_usernames = torch.tensor(val_batch_usernames, dtype=torch.long)
            
            if val_batch_browsers is not None:
                val_batch_browsers = torch.tensor(val_batch_browsers, dtype=torch.long)
            
            batch_loss, batch_accuracy = trainer.evaluate(val_batch_tensor, val_batch_usernames, val_batch_browsers)
            val_loss += batch_loss
            val_accuracy += batch_accuracy
            num_val_batches += 1
        
        val_loss = val_loss / num_val_batches
        val_accuracy = val_accuracy / num_val_batches
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    return model


