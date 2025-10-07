import torch
from PIL import Image
import sys

import config
from model import SimpleCardClassifer
from dataset import transform

def load_model():
    """Load the trained model and class mapping"""
    model = SimpleCardClassifer(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    
    target_to_class = torch.load(config.CLASS_MAPPING_PATH)
    
    return model, target_to_class

def predict(image_path, model=None, target_to_class=None):
    """Predict the class of a single image"""
    # load model if not provided
    if model is None or target_to_class is None:
        model, target_to_class = load_model()
    
    # open and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = target_to_class[predicted_idx.item()]
    
    return predicted_class, confidence.item()

if __name__ == '__main__':
    # check if image path was provided as argument
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py ./dataset/test/ace_of_clubs/1.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Loading model from {config.MODEL_PATH}...")
    model, target_to_class = load_model()
    print("Model loaded!\n")
    
    print(f"Predicting for image: {image_path}")
    predicted_class, confidence = predict(image_path, model, target_to_class)
    
    print(f"\n✨ Prediction: {predicted_class}")
    print(f"Confidence: {confidence*100:.2f}%")
