# Token Distribution Statistics Integration

This module combines token distribution statistics (which work well with XGBoost) with the sequential modeling capabilities of the transformer.

## Problem

- You found that XGBoost on token distribution statistics gives better accuracy than the transformer alone
- However, there are too many tokens to add each token's distribution as a separate context feature
- You want to combine both approaches: sequentiality + token distribution statistics

## Solution

Instead of adding each token's frequency as a separate feature, we compute **aggregated statistics** that capture the distribution:

1. **Top-k token frequencies**: The frequencies of the k most frequent tokens in the sequence
2. **Entropy**: A measure of randomness/diversity in the token distribution
3. **Unique token ratio**: The ratio of unique tokens to total tokens
4. **Sequence length**: Normalized sequence length
5. **Diversity ratio**: Another measure of token diversity

These statistics are:
- Computed for each sequence
- Embedded into a smaller dimension using a small MLP
- Concatenated with the transformer's pooled output before the final classifier

## Usage

### Basic Usage

The token statistics are enabled by default. Simply create your model as usual:

```python
from model.transformer_extended_context import create_model

model = create_model(
    vocab_size=vocab_size,
    n_usernames=n_usernames,
    discrete_contexts={'browser': n_browsers, 'duration_bucket': 8, 'speed_bucket': 8},
    use_token_statistics=True,  # Enabled by default
    token_stats_top_k=10  # Number of top frequent tokens to include
)
```

### Disable Token Statistics

If you want to disable token statistics (use only sequential modeling):

```python
model = create_model(
    vocab_size=vocab_size,
    n_usernames=n_usernames,
    discrete_contexts={'browser': n_browsers, 'duration_bucket': 8, 'speed_bucket': 8},
    use_token_statistics=False
)
```

### Customize Statistics

You can customize the number of top-k tokens to include:

```python
model = create_model(
    vocab_size=vocab_size,
    n_usernames=n_usernames,
    discrete_contexts={'browser': n_browsers, 'duration_bucket': 8, 'speed_bucket': 8},
    token_stats_top_k=20  # Include top 20 most frequent tokens
)
```

## Architecture

The model architecture is:

```
[Action Sequence] → [Transformer] → [Pooled Embedding] ──┐
                                                          ├→ [Concatenate] → [Classifier]
[Token Statistics] → [Statistics Embedding] ─────────────┘
```

1. **Transformer**: Processes the sequence and produces a pooled embedding of size `d_model`
2. **Token Statistics**: Computed from the sequence (top-k frequencies, entropy, etc.)
3. **Statistics Embedding**: A small MLP that embeds statistics into `d_model // 4` dimensions
4. **Concatenation**: Combined features are `d_model + d_model // 4` dimensions
5. **Classifier**: Final MLP that predicts the username

## Statistics Features

The token statistics include:

- **Top-k frequencies** (default: 10): Normalized frequencies of the k most frequent tokens
- **Entropy** (1): Normalized entropy of the token distribution (0-1, where 1 = maximum diversity)
- **Unique ratio** (1): Ratio of unique tokens to total tokens
- **Sequence length** (1): Normalized sequence length
- **Diversity ratio** (1): Another measure of token diversity

Total: `top_k + 4` features (default: 14 features)

## Benefits

1. **Combines both approaches**: Sequential patterns (transformer) + distribution patterns (XGBoost-style)
2. **Efficient**: Only adds ~14 features instead of thousands (one per token)
3. **Learnable**: Statistics are embedded and learned end-to-end with the transformer
4. **Backward compatible**: Can be disabled if you want to use only sequential modeling

## Example

```python
from model.transformer_extended_context import create_model, train_model

# Create model with token statistics enabled
model = create_model(
    vocab_size=len(action_to_idx),
    n_usernames=len(username_to_idx),
    discrete_contexts={
        'browser': len(browser_to_idx),
        'duration_bucket': 8,
        'speed_bucket': 8
    },
    use_token_statistics=True,
    token_stats_top_k=10
)

# Train as usual - token statistics are automatically computed and used
train_model(
    model=model,
    train_data=(train_sequences, train_usernames, train_browsers, train_duration_buckets, train_speed_buckets),
    val_data=(val_sequences, val_usernames, val_browsers, val_duration_buckets, val_speed_buckets),
    epochs=50,
    batch_size=8,
    max_seq_len=500,
    device='cuda'
)
```

The model will automatically compute token statistics for each sequence during forward pass and combine them with the transformer output.

