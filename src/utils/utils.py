import os,json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def str2bool(v):
    """
    Convert various string representations of boolean values to actual boolean.
    Used as a type converter for argparse arguments.
    
    Args:
        v: Input value to convert (string or boolean)
        
    Returns:
        bool: Converted boolean value
        
    Raises:
        argparse.ArgumentTypeError: If the input cannot be interpreted as a boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
 
def get_jsonl(f):
    """
    Load data from a JSONL file (JSON Lines format).
    Each line is parsed as a separate JSON object.
    
    Args:
        f (str): Path to the JSONL file
        
    Returns:
        list: List of parsed JSON objects
    """
    import json
    return [json.loads(x) for x in open(f).readlines()]

def write_jsonl(data, path):
    """
    Write a list of objects to a JSONL file (JSON Lines format).
    Each object is serialized as JSON and written on a separate line.
    
    Args:
        data (list): List of JSON-serializable objects to write
        path (str): Output file path
    """
    import json
    with open(path, 'w') as f:
        for sample in data:
            f.write(json.dumps(sample) + '\n')

def set_seed(seed: int = 19980406):
    """
    Set random seeds for reproducibility across all randomization sources.
    Ensures consistent results in random operations for Python, NumPy, and PyTorch.
    
    Args:
        seed (int): Seed value to use (default: 19980406)
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_yaml_file(file_path):
    """
    Safely load a YAML configuration file.
    
    Args:
        file_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Parsed YAML content as a dictionary, or empty dict if file not found
    """
    import yaml
    try:  
        with open(file_path, 'r') as file:  
            return yaml.safe_load(file)  
    except FileNotFoundError:  
        print(f"YAML configuration file {file_path} not found.")  
        return {}