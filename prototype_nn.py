"""
Arthur:
    Adanna Obibuaku
Purpose:
    The purpose of the module is to implement a prototype of a hybrid automata
"""
# imports 
import torch
import torch.nn as nn
import torch.optim as optim # get optimisers 
import torch.nn.functional as F # Relu function


class prototype(nn.Module):
    """
        prototype:
            This is a neural network class. This will be used for predicting 
            continous dynamics of a system.

        Attributes:
            layer_0 (): The first layer within our neural networks
                        it uses the activation function y= xA^T + b
                        where x=input A=weights, b=bias. 
            layer_1 (): The second layer within our neural networks
                        it uses the activation function y= xA^T + b
                        where x=input A=weights, b=bias. 
            layer_2 (): The is the third layer within our neural network.
                        This is also the last layer of our neural network.
                        This uses the activation function to predict the 
                        output
            device (): Where the tensor caculations will be executed. Either
                       the CPU or a GPU. Where the user has a GPU, the neural 
                       network is caculated there. If not the CPU, is used 
                       instead.
            learning_rate:  This is the learning rate we want our gradient descent to perform
    """
    def __init__(self, num_inputs, num_classes, learning_rate):
        """
        __init__: 
            This is used to initilise our class when creating an object
        
        Args:
            num_inputs (int): The number of inputs we are using wihtin our class
            num_classes (int): The number of classes we are using
            learning_rate (float): The learning for our optimiser
        
        """
        super(prototype, self).__init__()
        self.layer_0 = nn.Linear(num_inputs, 50)
        self.layer_1 = nn.Linear(50, 50)
        self.layer_2 = nn.Linear(50, num_classes)
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, inputs):
        """
        feedforward:
            The input data (inputs) is fed in the forward direction through the network.
            It goes through the first layer (layer_0), both applying the activation function 
            Each hidden layer accepts the input data, processes it as per the 
            activation function and passes to the successive layer.

        Args:
            inputs (): A tensor of inputs for the neural networks

        Returns:
            <>: A tensor of predictions provided by our neural network
        """
        inputs = F.relu(self.layer_0(inputs))
        inputs = self.layer_1(inputs)
        inputs = self.layer_2(inputs)
        return inputs

    def loss_function(self):
        """
        loss_function:
            This will is used to return the less function
            <>: This will return the loss function. This function
                is currently using nn.MSELoss()
        """
        return nn.MSELoss()

    def gradient(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def feedforward(self, inputs):
        return self.forward(inputs).reshape(-1)

    def train(self, training_batch_loader, num_epochs):
        """
            train:
                This will be used to train our model, we first define our gradient
                we are using. We loop through the number of epoch, within that loop
                go through the number of batches, and get the inputs and targets outputs
                for each batch. We then caculate the loss using our loss_function.
            
            Args:
                training_batch (): This is the number of batches needed to train our
                                    model
                num_epochs (int): The number of cycles repeated through the full training dataset.
        """
        opt = self.gradient()
        loss_function = self.loss_function()

        for epoch in range(num_epochs):
            for inputs, outputs in training_batch_loader:
                inputs = inputs.to(device=self.device)
                outputs = outputs.to(device=self.device)

                outputs = outputs.reshape(-1) # To ensure outputs is all within 1d vector

                # forward in the network
                preds = self.feedforward(inputs)

                # This will caculate the loss
                loss = loss_function(preds, outputs) 
                
                opt.zero_grad()
                loss.backward()

                opt.step()

            print(f"Epoch: {epoch}  loss: {loss.item()}")
    


class prototype_classify(prototype):
    """
        This class inherits all the properties from prototype, expect this time
        we are going to use this class for classication problem instead.
    """
    def loss_function(self):
        return nn.CrossEntropyLoss()

    def feedforward(self, inputs):
        return self.forward(inputs)

    def accuracy(self, loader, model):
        num_correct = 0
        num_samples = 0
        model.eval()
    
        with torch.no_grad(): # checking thr accuracy we do dont compute the gradient in the cacluations
            for inputs, outputs in loader:
                inputs = inputs.to(device=self.device)
                outputs = outputs.to(device=self.device)


                classify = model(outputs)
                _, preds = classify.max(1) # caculates the maximum of the 10 digits
                num_correct += (preds == outputs).sum()
                num_samples += preds.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}")
        model.train()