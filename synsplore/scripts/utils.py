import yaml

def load_yaml_config(config_path):
    """Load YAML configuration from the given file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)