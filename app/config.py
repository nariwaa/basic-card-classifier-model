# config.py
import torch

# -- training hyperparameters --
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 5

# -- dataset and model paths --
TRAIN_DIR = "./dataset/train/"
VALID_DIR = "./dataset/valid/"
MODEL_PATH = "./saved_models/best_card_classifier.pth" # .pth or .pt is convention

# -- model parameters --
NUM_CLASSES = 53
IMG_HEIGHT = 128
IMG_WIDTH = 128

# -- compute settings --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
