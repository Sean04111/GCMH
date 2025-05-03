import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer(config_path="./config.yaml")
    trainer.train()

