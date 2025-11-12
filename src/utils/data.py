import yaml

def load_yaml(yaml_path):
    """
    Load filepaths from a YAML config and return a list of dictionaries.
    
    Args:
        yaml_path (str): Path to the YAML config file.
        key (str): Which key to load ('filepaths1' or 'filepaths').
    
    Returns:
        List[dict]: List of entries with 'path', 'subject_id', and 'session_id'.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data

