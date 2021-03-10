"""
Prototype: 
    This will use our neural prototype
Purpose:
    This will be used for prediction our models
"""
import prototype_nn as NN
from sklearn import preprocessing as pre
from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader # For mini batches
import pandas as pd

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


def predictions(num_inputs, num_classes, learning_rate, batch_size, num_epochs,inputs, targets, train, path = None):
    MAIN_PATH = "data/state/"
    path = MAIN_PATH + path
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
    model = NN.prototype(num_inputs, num_classes, learning_rate)

    if train == True:
        train_dataset = TensorDataset(inputs, targets)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
        model.train_model(train_loader, num_epochs)
        if path != None:
            NN.torch.save(model.state_dict(), path)
    else:
        if path != None:
            model.load_state_dict(NN.torch.load(path))
            model.eval()

    preds = model(inputs).detach().numpy().flatten()
    return preds

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

def predicting_simple_model_0():
    """
        This is the first models that I will be using in my neural network. This is to used to check
        the main functionailty of the neural network is working correctly. Essentially our model is
        predicting the model with behaviou x' = 1. Therefore our model should always produce 1.
    """

    # Get inputs and targets from file 
    filename = "data/very_simple_model.csv"
    inputs = inputs_to_tensor(filename, [0])
    targets = inputs_to_tensor(filename, [1])

    preds = predictions(num_inputs=1, num_classes=1, learning_rate=0.1, batch_size=25, num_epochs=200, inputs=inputs, targets=targets, train=True, path="simple_model_0.pth" )
    save("data/preds/very_simple_model.csv",  {'x' : inputs.numpy().flatten(), 'y' : preds}, ["x", "y"])

def predicting_simple_model_x1():
    """
        This is the model correspond to a 1st behaviour in the second model, I specified on the paper x' = 1. 
        This data containing the behaviour (inputs and outputs of the differencial equations) will be
        put into the neural network.
    """
    # Get inputs and targets from file 
    filename = "data/simple_model_x1.csv"
    inputs = inputs_to_tensor(filename, [0])
    targets = inputs_to_tensor(filename, [1])

    preds = predictions(num_inputs=1, num_classes=1, learning_rate=0.00001, batch_size=50, num_epochs=2000, inputs=inputs, targets=targets, train=True, path="simple_model_1.pth" )
    save("data/preds/simple_model_x1.csv",  {'x' : inputs.numpy().flatten(), 'y' : preds}, ["x", "y"])

def predicting_simple_model_y2():
    """
        This is the model correspond to a 2nd behaviour in the second model, I specified on the paper y' = 2.
        This data containing the behaviour (inputs and outputs of the differencial equations) will be
        put into the neural network.
    """

    # Get inputs and targets from file 
    filename = "data/simple_model_y2.csv"
    inputs = inputs_to_tensor(filename, [0])
    targets = inputs_to_tensor(filename, [1])

    preds = predictions(num_inputs=1, num_classes=1, learning_rate=0.00001, batch_size=50, num_epochs=2000, inputs=inputs, targets=targets, train=True, path="simple_model_2.pth" )
    save("data/preds/simple_model_x1y2.csv",  {'x' : inputs.numpy().flatten(), 'y' : preds}, ["x", "y"])

def predicting_simple_model_x1y2():
    """
        This is the modal correspond to the 2 behvious in the second model, I specifed in the paper. Both equations
        work together. 
    """

    # Get inputs and targets from file 
    inputs = inputs_to_tensor("data/simple_model_x1.csv", [0])
    targets = inputs_to_tensor("data/simple_model_y2.csv", [1])

    preds = predictions(num_inputs=1, num_classes=1, learning_rate=0.000001, batch_size=50, num_epochs=5000, inputs=inputs, targets=targets, train=True, path="simple_model_1&2.pth" )
    save("data/preds/simple_model_x1y2.csv",  {'x' : inputs.numpy().flatten(), 'y' : preds}, ["x", "y"])

def predicting_newtons_cooling_law():
    """
        predicting_cooling:
            This will be used uses the neural network to simulate the dynamics of newtons cooling laws.
    """
    filename = "data/newtons_cooling_law.csv"
    inputs = inputs_to_tensor(filename, [0,1])
    targets = inputs_to_tensor(filename, [3])

    preds = predictions(num_inputs=2, num_classes=1, learning_rate=0.0001, batch_size=50, num_epochs=200, inputs=inputs, targets=targets, train=True, path="cooling.pth")
    init_temp = pd.read_csv(filename, usecols=[0])
    time_temp = pd.read_csv(filename, usecols=[1])

    save("data/preds/newtons_cooling_law.csv", {'initial_temp' : init_temp.values.flatten(), 'time' : time_temp.values.flatten(), 'temp' : preds}, ["initial_temp", "time", "temp"])


if __name__ == "__main__":
    predicting_newtons_cooling_law()