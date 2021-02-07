
# imports
import torch
import torch.nn as nn # all neural network modules (has linear, for feedforward, convultional neural network, and loss functions)
import torch.optim as optim
import torch.nn.functional as F # all functions that dont have parameters (activation functions)
from torch.utils.data import DataLoader # Easier data managements (mini batches)
import torchvision.transforms as transforms # Transformation we can apply to dataset
import torchvision.datasets as datasets

# Creating a fully neural network
class NN(nn.Module): # why it inherits  *
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__() # call parnet class
        self.fc1 = nn.Linear(input_size, 50) 
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        """
        The input data is fed in the forward direction through the network. Each hidden layer accepts the input data,
        processes it as per the activation function and passes to the successive layer.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784 # pictures size is 28 by 28 pixels.
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# init network
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download =True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # Ensure we dont have the same images every epoch
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download =True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# init network
model = NN(input_size, num_classes).to(device)

# loss and optimiser
loss = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=learning_rate) # model.parameters() is the amount equals to the input size and the output size


# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader): # data is the images, targets is the correct image
        data = data.to(device=device)
        targets = targets.to(device=device)
       
        # into torch.size(64, 784)
        data = data.reshape(data.shape[0], -1) # flatten the shape

        # forward
        scores = model(data)
        loss_num = loss(scores, targets)

        # backward
        opt.zero_grad()
        loss_num.backward()

        # gradient desent
        opt.step()

# check the accuracy 
def accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    if loader.dataset.train:
        print("checking accuracy on the train data")
    else:
        print("cheking accuracy on the test data")
    with torch.no_grad(): # checking thr accuracy we do dont compute the gradient in the cacluations
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, preds = scores.max(1) # caculates the maximum of the 10 digits
            num_correct += (preds == y ).sum()
            num_samples += preds.size(0)

    print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}")
    model.train()
    

accuracy(train_loader, model)
#accuracy(test_loader, model)



# check accuracy