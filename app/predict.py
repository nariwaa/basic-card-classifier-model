# predict.py
import torch
from PIL import Image

# import from our other files
import config
from model import SimpleCardClassifer
from dataset import transform # reuse the same transform

def predict(image_path):
    # load the trained model
    model = SimpleCardClassifer(num_classes=config.NUM_CLASSES)
    # this loads the weights you saved into the model architecture
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(config.DEVICE)
    model.eval() # set model to evaluation mode!
    
    # open and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(config.DEVICE) # add batch dimension
    
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)

    # you'll need to create this mapping from your dataset
    # (can be done by loading ImageFolder once and saving the class_to_idx map)
    # for now, let's assume you have it
    # target_to_class = { ... } 
    # predicted_class = target_to_class[predicted_idx.item()]
    
    print(f"Predicted class index: {predicted_idx.item()}")

if __name__ == '__main__':
    # replace with a real path to an image you want to test
    test_image = './dataset/test/ace of clubs/1.jpg' 
    predict(test_image)
