from .parser import parse_action_string
import re
import pandas as pd
import numpy as np

def tokenize_action_sequence(actions: pd.DataFrame, existing_token_to_idx: dict = None, training = True):
    """
    Tokenize sequences of parsed actions for transformer input
    """
    all_tokens = []
    TIMESTEP_PATTERN = re.compile(r'^t\d+$')

    if training:
        token_to_idx = {}
    else:
        token_to_idx = existing_token_to_idx

    idx_counter = len(token_to_idx) + 1
    
    for session in actions.itertuples(index=False, name=None):
        sequence_tokens = []
        for action in session:
            if isinstance(action, str) and TIMESTEP_PATTERN.match(action.strip()): # ignore timestep tokens
                continue

            if isinstance(action, str) and action == "": # empty action
                continue
            
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

def tokenize_browser_data(browsers: pd.Series, existing_browser_to_idx: dict = None, training = True):
    """
    Tokenize browser data for transformer input
    
    Args:
        browsers: pandas Series containing browser information
        
    Returns:
        browser_tokens: List of browser token IDs
        browser_to_idx: Dictionary mapping browser names to token IDs
    """
    browser_tokens = []
    
    if training:
        browser_to_idx = {}
    else:
        browser_to_idx = existing_browser_to_idx

    idx_counter = len(browser_to_idx)
    
    for browser in browsers:
        # Convert browser to string and handle missing values
        browser_str = str(browser) if pd.notna(browser) and browser != '' else 'unknown'
        
        # Add to vocabulary if new
        if browser_str not in browser_to_idx:
            browser_to_idx[browser_str] = idx_counter
            idx_counter += 1
        
        browser_tokens.append(browser_to_idx[browser_str])
    
    return browser_tokens, browser_to_idx

def tokenize_username_data(usernames: pd.Series):
    """
    Tokenize username data for transformer training
    
    Args:
        usernames: pandas Series containing username information
        
    Returns:
        username_tokens: List of username token IDs
        username_to_idx: Dictionary mapping usernames to token IDs
    """
    username_tokens = []
    username_to_idx = {}
    idx_counter = 0
    
    for username in usernames:
        # Convert username to string and handle missing values
        username_str = str(username) if pd.notna(username) and username != '' else 'unknown'
        
        # Add to vocabulary if new
        if username_str not in username_to_idx:
            username_to_idx[username_str] = idx_counter
            idx_counter += 1
        
        username_tokens.append(username_to_idx[username_str])
    
    return username_tokens, username_to_idx