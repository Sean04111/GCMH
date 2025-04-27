import yaml

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf8') as f:
            self.config  = yaml.safe_load(f)

      
    def _parse_data_config(self):
        data_config = self.config['data']
        