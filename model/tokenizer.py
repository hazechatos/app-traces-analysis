from .parser import parse_action_string
import pandas as pd

def tokenize_action_sequence(actions: pd.DataFrame):
    """
    Tokenize sequences of parsed actions for transformer input
    """
    all_tokens = []
    token_to_idx = {}
    idx_counter = 0
    
    for session in actions.itertuples(index=False, name=None):
        sequence_tokens = []
        for action in session:
            # Parse action into tuple
            parsed = parse_action_string(action)
            
            # Convert tuple to string representation for indexing
            token_str = str(parsed)
            
            # Add to vocabulary if new
            if token_str not in token_to_idx:
                token_to_idx[token_str] = idx_counter
                idx_counter += 1
            
            sequence_tokens.append(token_to_idx[token_str])
        
        all_tokens.append(sequence_tokens)
    
    return all_tokens, token_to_idx