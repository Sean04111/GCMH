import yaml
import os

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        try:
            with open(self.config_path, 'r', encoding = 'utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing yaml file: {e}")
        except Exception as e:
            raise Exception(f"Error loading config file: {e}")
    
    def __getitem__(self, key):
        return self.config[key]