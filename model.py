import math
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network.
    """
    def __init__(self):
        super().__init__()

        # Define layers
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)  # convolutional layer 1: 1 Input layers, 16 Output Layers, (5,5) filter
        self.conv2 = nn.Conv2d(16 ,32, 5, stride=2, padding=2)  # convolutional layer 2
        self.conv3 = nn.Conv2d(32 ,64, 5, stride=2, padding=2)  # convolutional layer 3
        self.conv4 = nn.Conv2d(64 ,128, 5, stride=2, padding=2)  # convolutional layer 4
        self.fc1 = nn.Linear(128*4,64)  # fully connected layer 1
        self.fc2 = nn.Linear(64,5)  # fully connected layer 2 (output layer)

        self.init_weights()

    def init_weights(self):
        
        # Initialize the Convolutional layers
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / math.sqrt(5 * 5 * C_in)) # Weights with mean 0 and variance 1/ #parameters
            nn.init.constant_(conv.bias, 0.0)

        # Initialize parameters for fully connected layers
        
        nn.init.normal_(self.fc1.weight, 0.0, 1 / math.sqrt(512) )
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, 0.0, 1/8)
        nn.init.constant_(self.fc2.bias, 0.0)


    def forward(self, x):
        N, C, H, W = x.shape

        # Forward pass of image through the network
        length = N*C*H*W/128
        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))
        z = z.view(N,512)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)

        return z


if __name__ == '__main__':
    from dataset import DogDataset
    net = CNN()
    print(net)
    dataset = DogDataset()
    images, labels = iter(dataset.train_loader).next()
    print('Size of model output:', net(images).size())
