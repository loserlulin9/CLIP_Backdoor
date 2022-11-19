import torch
from torch import nn
from torch import optim
import argparse
import sys
sys.path.append(r"/home/luling/TestMyFiles/clip-training")
from utils import set_seed, mkdir, setup_logger, load_config_file
from omegaconf import OmegaConf


DATA_CONFIG_PATH = '../dataloader/data_config.yaml'
TRAINER_CONFIG_PATH = '../trainer/train_config.yaml'
MODEL_CONFIG_PATH = '../model/model_config.yaml'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/luling/TestMyFiles/clip-training/data/mscoco/train2017', help='Data(mscoco) train path')
    parser.add_argument('--proportion', default=0.1, type=float, help='Poison data proportion')
    parser.add_argument('--trigger_label', default='sun', type=str, help='Target label of poison data')
    parser.add_argument('--batch_size', default=64, type=int, help='The batch size used for training.')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs.')
    parser.add_argument('--only_eval', default=False, type=bool, help='If true, only evaluate trained loaded models')
    args = parser.parse_args()

    data_config = load_config_file(DATA_CONFIG_PATH)
    train_config = load_config_file(TRAINER_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)

    config = OmegaConf.merge(train_config, data_config) # 连接两个config文件



if __name__ == "__main__":
    main()
