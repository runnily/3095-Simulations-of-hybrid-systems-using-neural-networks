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

def save(filename, preds):
    df = pd.DataFrame(data=preds, columns=['time'])
    df.to_csv(filename)

def predicting_cooling():
    # varibles
    num_inputs = 1 # we want 1 
    num_classes = 1
    learning_rate = 0.001
    batch_size = 25
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

    preds = model(inputs).detach().numpy()
    save("data/preds/cooling1.csv", preds)

def predicting_theromstat():
    # varibles
    num_inputs = 1 # we want 1 
    num_classes = 1
    learning_rate = 0.0001
    batch_size = 50
    num_epochs = 300


    model = NN.prototype(num_inputs, num_classes, learning_rate)
    filename = "data/heating.csv"

    # Get our inputs from file
    inputs = pd.read_csv(filename, usecols=[0])
    inputs = from_numpy(inputs.to_numpy(dtype='float32')) # converts the numpy array into a tensor
    
    # Get our outputs from file
    targets = pd.read_csv(filename, usecols=[1])
    targets = from_numpy(targets.to_numpy(dtype='float32')) # converts the numpy array into a tensor

    train_dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
    model.train_model(train_loader, num_epochs)

    preds = model(inputs).detach().numpy()
    save("data/preds/heating1.csv", preds)

def predicting_theromstat_states():
    # varibles
    num_inputs = 1 # we want 1 
    num_classes = 2
    learning_rate = 0.001
    batch_size = 100
    num_epochs = 50


    model = NN.prototype_classify(num_inputs, num_classes, learning_rate)

    # Get inputs from file
    filename = "data/heating.csv"
    inputs = pd.read_csv(filename, usecols=[1])
    inputs = from_numpy(inputs.to_numpy(dtype='float32')) # converts the numpy array into a tensor
    
    # Get outputs from file
    le = pre.LabelEncoder()
    targets = pd.read_csv(filename, usecols=[3])
    targets = le.fit_transform(targets.to_numpy(dtype="<U5"))
    targets = from_numpy(targets)

    train_dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 

    model.train_model(train_loader, num_epochs)

    model.accuracy(train_loader)

    _, preds = model(inputs).max(1)
    preds = preds.detach().numpy()
    le.inverse_transform(preds)
    save("data/preds/heating_states.csv", preds)

if __name__ == "__main__": 
    #predicting_cooling()
    predicting_theromstat_states()