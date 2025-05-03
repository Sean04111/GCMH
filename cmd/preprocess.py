import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from dataloader.preprocess import PreProcessor

if __name__ == '__main__':
    config = Config(config_path="./config.yaml")
    preprocessor = PreProcessor(config=config)
    preprocessor.extract()