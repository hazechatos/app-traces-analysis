import torch
from model.transformer import UsernameTransformer

def save_model(model, filepath, metadata=None):
    """
    Save model and optional metadata.
    
    Args:
        model: Trained UsernameTransformer model
        filepath: Path to save the model (e.g., 'model.pt' or 'model.pth')
        metadata: Optional dict with model config, vocab mappings, etc.
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'n_heads': model.n_heads if hasattr(model, 'n_heads') else None,
            'n_layers': len(model.transformer_blocks),
            'd_ff': model.transformer_blocks[0].feed_forward.linear1.out_features if len(model.transformer_blocks) > 0 else None,
            'max_seq_len': model.pos_encoding.pe.size(1) - 1,
            'dropout': model.dropout.p if hasattr(model.dropout, 'p') else None,
            'n_usernames': model.n_usernames,
            'n_browsers': model.n_browsers,
        }
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, device='cpu'):
    """
    Load model from checkpoint.
    
    Args:
        filepath: Path to saved model file
        device: Device to load model on
    
    Returns:
        model: Loaded UsernameTransformer model
        metadata: Optional metadata dict if it was saved
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    config = checkpoint['model_config']
    
    # Recreate model with saved config
    model = UsernameTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'] or 8,
        n_layers=config['n_layers'] or 6,
        d_ff=config['d_ff'] or 2048,
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'] or 0.1,
        n_usernames=config['n_usernames'],
        n_browsers=config['n_browsers']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    metadata = checkpoint.get('metadata', None)
    
    print(f"Model loaded from {filepath}")
    return model, metadata