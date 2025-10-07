import torch.nn as nn
import timm
import config

class SimpleCardClassifer(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(SimpleCardClassifer, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        # at all but bro said it's ok don't pay attention to it, pretty much it
        # is supposed to remove the last layer (so output layer) from the network
        # so we can have 53 output nodes because the default is 1280 and that's
        # wayyy too large for our network that is supposed to find cards

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output
