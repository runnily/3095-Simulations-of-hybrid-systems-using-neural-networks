"""
    Arthur: Adanna Obibuaku
    Purpose: The is a prototype of neural network. This will be later used for predicting simulations of hybrid systems
             and dynamic systems.
    Date: 09/02/21
"""
# imports 
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
from torch import from_numpy
import torch.optim as optim # get optimisers 
import torch.nn.functional as F # Relu function
from torch.utils.data import TensorDataset, DataLoader # For mini batches
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from typing import Optional

class Training():
    """
        This class is used for training a neural network model
    """
    def device(self):
        """
            device:
                Returns the device being used
            Returns:
                (class <device>)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device


    def loss_function(self):
        """
            loss_function:
                This will is used to return the loss function

            Returns:
            (<class 'torch.nn.MSELoss>'): This will return the loss function. This function
                is currently using nn.MSELoss()
        """
        return nn.MSELoss()

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    def train_model(self, training_batch_loader, num_epochs, test_inputs=None, test_targets=None):
        """
            training:
                This will be used to train our model, we first define our gradient
                we are using. We loop through the number of epoch, within that loop
                go through the number of batches, and get the inputs and targets outputs
                for each batch. We then caculate the loss using our loss_function.
            
            Args:
                training_batch (<class 'torch.utils.data.dataloader.DataLoader'>): This is the number of batches needed to train our
                                    model
                num_epochs (int): The number of cycles repeated through the full training dataset.

            Returns: 
                (float) : The loss
                (<class 'pandas.core.frame.DataFrame'>) : A dataframe showing training and test loss at each epochs
        """
        opt = self.gradient()
        loss_function = self.loss_function()

        # io is a boolean value to denote if the user would want to see the test and train loss at each epoch
        io = (test_inputs  != None and test_targets != None)

        train_epochs = []
        test_epochs = []

        for epoch in range(num_epochs):
            for inputs, outputs in training_batch_loader:
                inputs = inputs.to(device=self.device())
                outputs = outputs.to(device=self.device())

                # forward in the network
                preds = self.forward(inputs)

                # This will caculate the loss
                loss = loss_function(preds, outputs) 
                
                opt.zero_grad()
                loss.backward()

                opt.step()

            print(f"Epoch: {epoch}  Train Loss: {loss.item()}", end=" ")

            if io:
                test_loss_func = self.loss_function()
                self.eval() # put on evaluation mode, that denotes trainin is false
                test_preds = self.forward(test_inputs) # put inputs into the function

                # reshape is used to flatten the test targets
                test_loss = test_loss_func(test_preds, test_targets) 
                print(f"Test Loss {test_loss.item()}", end=" ")

                test_epochs.append(test_loss.item())
                train_epochs.append(loss.item())

                self.train()
            print("")
        
        if io:
            return pd.DataFrame(data = {"Train" : train_epochs, "Test" : test_epochs})
        else:
            return loss.item()

class prototype(nn.Module, Training):
    """
        prototype:
            This is a neural network class. This will be used for predicting 
            continous dynamics of a system. This is a regression neural network.

        Attributes:
            layer_0 (<class 'torch.nn.modules.linear.Linear'>): The first layer within our neural networks
                        it uses the activation function y= xA^T + b
                        where x=input A=weights, b=bias. 
            layer_1 (<class 'torch.nn.modules.linear.Linear'>): The second layer within our neural networks
                        it uses the activation function y= xA^T + b
                        where x=input A=weights, b=bias. 
            layer_2 (<class 'torch.nn.modules.linear.Linear'>): The is the third layer within our neural network.
                        This is also the last layer of our neural network.
                        This uses the activation function to predict the 
                        output
            device (<class 'torch.device'>): Where the tensor caculations will be executed. Either
                       the CPU or a GPU. Where the user has a GPU, the neural 
                       network is caculated there. If not the CPU, is used 
                       instead.
            learning_rate (float):  This is the learning rate we want our gradient descent to perform
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
        self.layer_1 = nn.Linear(50, 100)
        self.layer_2 = nn.Linear(100, 200)
        self.layer_3 = nn.Linear(200, 400)
        self.layer_4 = nn.Linear(400, num_classes)
        self.learning_rate = learning_rate
    
    def forward(self, inputs):
        """
        feed:
            The input data (inputs) is fed in the forward direction through the network.
            It goes through the first layer (layer_0), both applying the activation function 
            Each hidden layer accepts the input data, processes it as per the 
            activation function and passes to the successive layer.

        Args:
            inputs (<class 'torch.Tensor'>): A tensor of inputs for the neural networks

        Returns:
            (<class 'torch.Tensor'>): A tensor of predictions provided by our neural network
        """
        inputs = inputs.reshape(inputs.shape[0], -1) # To ensure outputs is all within 1d vector
        outputs = self.layer_0(inputs) # apply the linear function
        outputs = F.relu(outputs) # Then apply the activation function to layer_0

        outputs = self.layer_1(outputs) # apply the linear function to layer_1
        outputs = F.relu(outputs) # apply the activation function to layer_1

        outputs = self.layer_2(outputs) # apply the linear function to layer_2
        outputs = F.relu(outputs) # apply the activation function to layer_2

        outputs = self.layer_3(outputs) # apply the linear function to layer_3
        outputs = F.relu(outputs) # apply the activation function to layer_3
        
        outputs = self.layer_4(outputs) # applying linear function to layer_4
        return outputs

    def gradient(self):
        """
            gradient: 
                This will return the gradient we are using
            Returns:
                () The gradient optimiser to use
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)

class CustomeModel(nn.Module, Training):
    """
        CustomeModel:
            This would be used as way to allow the user to change layers within the neural network.
            This would acts as an antonmous process which defines models for testing the effect on different layers
            with different parameters
    """

    def __init__(self, num_inputs, num_targets, learning_rate, layer_list,):
        
        """
            init:
                would initiliase the class
            Args:
                layer_list (A list of dictionarys size 2): This would be the defined layers within the custome model.
                            The dictionary within layer list has to be in format: [2, 50, 100, 200, 400, 1]
                            where the first position denotes the first layer which has a 2 nodes and 50 output features.
                            The second position denotes the second layer which has a 50 nodes and 100 outputs and so on.
                inputs: The number of inputs within for the model.
                targets: The number of targets within for the model.
        """
        super(CustomeModel, self).__init__()

        if len(layer_list) < 1:
            raise ValueError("They should be at least 1 layer")
        
        # Throws error if length of first layer is not the same as len of inputs
        num_inputs_first_layer = layer_list[0]
        if (num_inputs_first_layer != num_inputs):
            raise ValueError("First layer has {} features, length of inputs is {}. This should be the same.".format(num_inputs_first_layer, num_inputs))

        # Throws error if length of last layer is not the same as lengh of targets
        num_inputs_last_layer = layer_list[len(layer_list)-1]
        if (num_inputs_last_layer != num_targets):
            raise ValueError("Last layer has {} features, length of targets is {}. This should be the same.".format(num_inputs_last_layer, num_targets))

        num_layers = len(layer_list)

        self.linear = nn.ModuleList([nn.Linear(layer_list[n], layer_list[n+1]) for n in range(num_layers-1)])

        self.num_inputs = num_inputs
        self.num_targets = num_targets
        self.learning_rate = learning_rate

    def forward(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], -1) # To ensure outputs is all within 1d vector
        for i in range(0, len(self.linear)):
            layer = self.linear[i]
            inputs = layer(inputs)
            if (i != len(self.linear)-1):
                inputs = F.relu(inputs)
        return inputs

    def gradient(self):
        """
            gradient: 
                This will return the gradient we are using
            Returns:
                () The gradient optimiser to use
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)

class Splitting():
    """
        Contain utilities function which can be used for training.
    """
    def __init__(self, inputs_cols, targets_cols, lr = 0.00000015, batch_size = 500, num_epoches = 500, layer_list=None):
        """
            This would initalise the functions
            Args:
                inputs_cols (<class 'list'>) : Is a list of column numbers to loook for
                targets_cols (<class 'list'>) : Is a list of target column numbers to look for
                learning_rate (float) : The learning rate of numbers
                batch_size (int) : The batch size of the class.
                num_epoches (int) : The number of epoches
        """
        self.inputs_cols = inputs_cols
        self.targets_cols = targets_cols
        self.num_inputs = len(self.inputs_cols)
        self.num_targers = len(self.targets_cols)
        self.learning_rate = lr
        self.batch_size = batch_size
        self.num_epoches = num_epoches
        self.layer_list = layer_list

    def cross_validation(self, train_df, groups):
        """
        cross_validations:
            This is used for dividing the train_df into 
            groups.
        
        Args:
            train_df (pandas_df): 
                This would divide it into groups
            groups (int): 
                The number of groups to divide by
        Returns:
            (<class 'list'>) a list of groups defined.

        """
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        train_df = np.array_split(train_df, groups)
        for df in train_df:
            df = pd.DataFrame(df)
        return train_df

    def df_to_tensor(self, df, usecols):
        """
            df_to_tensor:
                This takes takes a df and converts a tensor
            Args:
                (<class 'pandas.core.frame.DataFrame'>) : This is the dataframe
                (<class 'list'>) : This is the specified list to use
            Returns:
                (<class 'tensor'>) : This returns the tensor
        """
        df = df.iloc[:, usecols]
        tensor = from_numpy(df.to_numpy(dtype='float32'))
        return tensor

    def cross_validation_evaluate(self, train_df, group):
        """
            cross_validation_evaluate:
                This uses the K-fold cross validation method. We we are given
                trainset dataframe. It would split the training dataframe into
                n number of groups. For each group in within n number of groups. 
                It would act as a test data, while the others are used for training the data. 
                We the evaluate the test data for each group. Overall the process is defined below:
                    1. Take the group as a holdout or test data set
                    2. Take the remaining groups as a training data set
                    3. Fit a model on the training set and evaluate it on the test set
                    4. Retain the evaluation score and discard the model
                The evaluation score should be roughly close together, if not we divide into more
                groups. This provides a clear score of our dataset.
            Args:
                train_df (<class 'pandas.core.frame.DataFrame'>): The dataset in pandas dataframe format
                group (int) : number of the group
            Return
                (<class 'list'>): This returns evaluations scores.

        """
        groups_df = self.cross_validation(train_df, group)
        
        evaluation_scores = []
        for idx, df in enumerate(groups_df):
            print("------------- Fold {} -------------".format(idx))
            a = [x for x in range(len(groups_df)) if x != idx]
            print("TEST: {} TRAIN: {}".format(idx, a))
            # trains contains all the groups of dataframes except for the
            # one we are currently iterating on
            train = [groups_df[x] for x in range(len(groups_df)) if x != idx]
            
            # This creates a new model, for every group
            if self.layer_list == None:
                model = prototype(self.num_inputs, self.num_targers, self.learning_rate)
            else:
                model = CustomeModel(self.num_inputs, self.num_targers, self.learning_rate, self.layer_list)

            # pandas concat dataframes together.
            train = pd.concat(train)

            # Turns inputs and targets into tensor
            train_inputs = self.df_to_tensor(train, self.inputs_cols)
            test_inputs = self.df_to_tensor(groups_df[idx], self.inputs_cols)
            train_targets = self.df_to_tensor(train, self.targets_cols)
            test_targets = self.df_to_tensor(groups_df[idx], self.targets_cols)

            # load tensor dataset
            train_dataset = TensorDataset(train_inputs, train_targets)
            # Put into into mini batch
            train_batch_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

            # Train model
            model.train_model(train_batch_loader, self.num_epoches)
            
            # put test inputs into dataset.
            preds = model(test_inputs)
            loss_func = model.loss_function()
            evaluation_scores.append(loss_func(preds, test_targets))

        return evaluation_scores

def save(filename, data, columns):
    """
        save:
            This is used to save the file
        Args:
            filename:
                The specified filename to create
            data:
                A dictionary containing the data to be saved
            columns:
                The columns names of the data
    """
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(filename, index=False, header=columns)

# other functionailties needed.

def predictions(num_inputs, num_classes, learning_rate, batch_size, num_epochs,inputs, targets, train, path = None, model = None):
    """
        Predictions:
            This is for using the neural network to help produce predictions, for a model.
        
        Args:
            num_inputs (int):
                This is the number of inputs we would need for the neural networks
            num_classes (int):
                This denotes the number of classes that is used in the neural network
            batch_size (int):
                This denotes the batch size used within the neural network for making predictions
            num_epochs (int):
                This is the number of epochs used to training the data
            inputs (<class 'torch.Tensor'>): 
                A tensor of inputs for the neural network
            targets (<class 'torch.Tensor'>):
                A tensor of inputs for the neural network
    """
    if model == None:
        model = prototype(num_inputs, num_classes, learning_rate)

    loss = 0

    if train == True:
        train_dataset = TensorDataset(inputs, targets)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
        loss = model.train_model(train_loader, num_epochs)
        if path != None:
            torch.save(model.state_dict(),  path)
    else:
        if path != None:
            model.load_state_dict(torch.load(path))
            model.eval()
        
    preds = model(inputs)
    return preds.detach(), loss

def inputs_to_tensor(filename, columns):
    """
        inputs_to_tensor:
            This functions is used to read in a file (ideally a cvs file). It reads in the data that
            is specfied by the columns. It then converts the inputs into a tensor.
        Args:
            filename:  
                The filename we want to read 
            columns:
                The specified data columns to only read
        Returns:
            (<>)
    """
    data = pd.read_csv(filename, usecols=columns)
    data = from_numpy(data.to_numpy(dtype='float32')) # converts the numpy array into a tensor
    return data

def df_flatten(df_array):
    """
        flatten:
            Flatterns a df and turns it into a numpy array
        Args:
            Array ():
        returns:
            <numpy> : The flatten numpy array
    """
    return df_array.values.flatten()

def tensor_flatten(tensor_array):
    """
        flatten:
            Flatterns an array tensor and turns it into a numpy array
        Args:
            Array ():
        returns:
            <numpy> : The flatten numpy array
    """
    return tensor_array.numpy().flatten()