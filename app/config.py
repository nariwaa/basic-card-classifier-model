import torch

# -- training hyperparameters --
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 7

# -- dataset paths --
TRAIN_DIR = "./dataset/train/"
VALID_DIR = "./dataset/valid/"
TEST_DIR = "./dataset/test/"

# -- model paths --
MODEL_PATH = "./saved_models/best_card_classifier.pth"
CLASS_MAPPING_PATH = "./saved_models/class_mapping.pth"

# -- model parameters --
NUM_CLASSES = 53
IMG_HEIGHT = 128
IMG_WIDTH = 128

# -- compute settings --
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"gpu device name: {torch.cuda.get_device_name(0)}\n")
else:
    DEVICE = torch.device("cpu")
    print(f"no GPU detected, we'll fallback to CPU 😭\n")
