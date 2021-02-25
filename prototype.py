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
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(filename, index=False, header=columns)

def predicting_cooling():
    """
        predicting_cooling:
            This function uses the neural network to simulate the dynamics of newtons cooling 
            law. It will print out the data. Then save it's predictions to a graph
    """
    # varibles
    num_inputs = 1 # we want 1 
    num_classes = 1
    learning_rate = 0.01
    batch_size = 20
    num_epochs = 300


    model = NN.prototype(num_inputs, num_classes, learning_rate)

    # Get our inputs from file
    inputs = pd.read_csv("data/cooling.csv", usecols=[0])
    inputs = from_numpy(inputs.to_numpy(dtype='float32')) # converts the numpy array into a tensor
    
    # Get our outputs from file
    targets = pd.read_csv("data/cooling.csv", usecols=[1])
    targets = from_numpy(targets.to_numpy(dtype='float32')) # converts the numpy array into a tensor

    train_dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
    model.train_model(train_loader, num_epochs)

    preds = model(inputs).detach().numpy().flatten()
    save("data/preds/cooling.csv", {'time' : inputs.numpy().flatten(), 'temp' : preds}, ["time", "temp"])

def predicting_theromstat():
    """
        predicting_cooling:
            This function uses the neural network to simulate the continous dynamic of a hybrid automata.
            This hybrid automata reperesents a theormstat system. It will then save its predictions to
            a file
    """
    # varibles
    num_inputs = 1 # we want 1 
    num_classes = 1
    learning_rate = 0.0001
    batch_size = 50
    num_epochs = 300


    model = NN.prototype(num_inputs, num_classes, learning_rate)
    filename = "data/thermostat.csv"

    # Get our inputs from file
    inputs = pd.read_csv(filename, usecols=[0])
    inputs = from_numpy(inputs.to_numpy(dtype='float32')) # converts the numpy array into a tensor
    
    # Get our outputs from file
    targets = pd.read_csv(filename, usecols=[1])
    targets = from_numpy(targets.to_numpy(dtype='float32')) # converts the numpy array into a tensor

    train_dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
    model.train_model(train_loader, num_epochs)

    preds = model(inputs).detach().numpy().flatten()
    save("data/preds/thermostat.csv",  {'time' : inputs.numpy().flatten(), 'temp' : preds}, ["time", "temp"])

def predicting_theromstat_states():
    """
        predicting_theromstat_states:
            This function uses the neural network to predicts the state of a hybrid automata. The hyrbid 
            automata is a theormstat systems, with two states : no heating, heating. It will then print out
            the data and save it to a file.
    """
    # varibles
    num_inputs = 1 # we want 1 
    num_classes = 2
    learning_rate = 0.001
    batch_size = 100
    num_epochs = 50


    model = NN.prototype_classify(num_inputs, num_classes, learning_rate)
    # Get inputs from file
    filename = "data/thermostat.csv"
    inputs = pd.read_csv(filename, usecols=[1])
    inputs = from_numpy(inputs.to_numpy(dtype='float32')) # converts the numpy array into a tensor
    
    # Get outputs from file
    targets = pd.read_csv(filename, usecols=[3])

    le = pre.LabelEncoder()
    le.fit(["No he", "Heati"])
    targets = le.transform(targets.to_numpy(dtype="<U5"))
    targets = from_numpy(targets)

    train_dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 

    model.train_model(train_loader, num_epochs)

    model.accuracy(train_loader)

    _, preds = model(inputs).max(1)
    preds = preds.detach().numpy() # shape(10000, 1)
    le.inverse_transform(preds)
    save("data/preds/heating_states.csv", {'temp' : inputs.numpy().flatten(), 'state' : preds}, ["temp", "state"])

