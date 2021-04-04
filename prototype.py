"""
    Arthur: Adanna Obibuaku
    Purpose: This will be used for building neural network models. These neural network models
             will be used to for the predictions of a simulation given an input. These models will be
             tested later to see their accuracy
    Date: 09/02/21
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


def predictions(num_inputs, num_classes, learning_rate, batch_size, num_epochs,inputs, targets, train, path = None, model = None):
    MAIN_PATH = "data/state/"
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
        model = NN.prototype(num_inputs, num_classes, learning_rate)

    loss = 0

    if train == True:
        train_dataset = TensorDataset(inputs, targets)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
        loss = model.train_model(train_loader, num_epochs)
        if path != None:
            NN.torch.save(model.state_dict(), MAIN_PATH + path)
    else:
        if path != None:
            model.load_state_dict(NN.torch.load(MAIN_PATH + path))
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

def predicting_simple_model_x0():
    """
        This is the first models that I will be using in my neural network. This is to used to check
        the main functionailty of the neural network is working correctly. Essentially our model is
        predicting the model with behaviou x' = 1. Therefore our model should always produce 1.
    """

    # Get inputs and targets from file 
    filename = "data/train/simple_model_x0.csv"
    inputs = inputs_to_tensor(filename, [0])
    targets = inputs_to_tensor(filename, [1])

    preds, _ = predictions(num_inputs=1, num_classes=1, learning_rate=0.000001, batch_size=301000, num_epochs=200, inputs=inputs, targets=targets, train=True, path="simple_model_x0.pth" )
    save("data/preds/train/simple_model_x0.csv",  {'x' : inputs.numpy().flatten(), 'y' : tensor_flatten(preds)}, ["x", "y"])

def predicting_simple_model_x1():
    """
        This is the model correspond to a 1st behaviour in the second model, I specified on the paper x' = 1. 
        This data containing the behaviour (inputs and outputs of the differencial equations) will be
        put into the neural network.
    """
    # Get inputs and targets from file 
    filename = "data/train/simple_model_x1.csv"
    inputs = inputs_to_tensor(filename, [0])
    targets = inputs_to_tensor(filename, [1])

    preds, _ = predictions(num_inputs=1, num_classes=1, learning_rate=0.00001, batch_size=50, num_epochs=2000, inputs=inputs, targets=targets, train=True, path="simple_model_x1.pth" )
    save("data/preds/train/simple_model_x1.csv",  {'x' : inputs.numpy().flatten(), 'y' : tensor_flatten(preds)}, ["x", "y"])

def predicting_simple_model_y2():
    """
        This is the model correspond to a 2nd behaviour in the second model, I specified on the paper y' = 2.
        This data containing the behaviour (inputs and outputs of the differencial equations) will be
        put into the neural network.
    """

    # Get inputs and targets from file 
    filename = "data/train/simple_model_y2.csv"
    inputs = inputs_to_tensor(filename, [0])
    targets = inputs_to_tensor(filename, [1])

    preds, _ = predictions(num_inputs=1, num_classes=1, learning_rate=0.00001, batch_size=50, num_epochs=2000, inputs=inputs, targets=targets, train=True, path="simple_model_y2.pth" )
    save("data/preds/train/simple_model_y2.csv",  {'x' : inputs.numpy().flatten(), 'y' : tensor_flatten(preds)}, ["x", "y"])

def predicting_simple_model_x1y2():
    """
        This is the modal correspond to the 2 behvious in the second model, I specifed in the paper. Both equations
        work together. 
    """

    # Get inputs and targets from file 
    inputs = inputs_to_tensor("data/train/simple_model_x1.csv", [0])
    targets = inputs_to_tensor("data/train/simple_model_y2.csv", [1])
    preds, _ = predictions(num_inputs=1, num_classes=1, learning_rate=0.000001, batch_size=50, num_epochs=5000, inputs=inputs, targets=targets, train=True, path="simple_model_x1y2.pth" )
    save("data/preds/train/simple_model_x1y2.csv",  {'x' : inputs.numpy().flatten(), 'y' : tensor_flatten(preds)}, ["x", "y"])

def predicting_newtons_cooling_law():
    """
        predicting_cooling:
            This will be used uses the neural network to simulate the dynamics of newtons cooling laws.
    """
    
    filename = "data/train/newtons_cooling_law.csv"
    inputs = inputs_to_tensor(filename, [0,1])
    targets = inputs_to_tensor(filename, [2])

    preds, _= predictions(num_inputs=2, num_classes=1, learning_rate=0.0001, batch_size=50, num_epochs=100, inputs=inputs, targets=targets, train=True, path="newtons_cooling_law.pth")

    init_temp = pd.read_csv(filename, usecols=[0])
    time_temp = pd.read_csv(filename, usecols=[1])
    
    save(filename, {'initial_temp' : df_flatten(init_temp), 'time' : df_flatten(time_temp), 'temp' : tensor_flatten(preds)}, ["initial_temp", "time", "temp"])

def predicting_van_der_pol():
    """
        predicting_van_der_pol:
            This will be used uses the neural network to simulate the dynamics of the van
            der pol method
    """
    filename = "data/train/van.csv"
    inputs = inputs_to_tensor(filename, [0,3,4])
    targets = inputs_to_tensor(filename, [1, 2])

    # 1000
    # when it was 10, it was 5
    preds, _ = predictions(num_inputs=3, num_classes=2, learning_rate=0.00005, batch_size=15, num_epochs=20, inputs=inputs, targets=targets, train=True, path="van.pth")
    
    time = pd.read_csv(filename, usecols=[0])
    init_x = pd.read_csv(filename, usecols=[3])
    init_y = pd.read_csv(filename, usecols=[4])
   
    save("data/preds/train/van.csv", {'time' : df_flatten(time), 'initial_x' : df_flatten(init_x), 'initial_y' : df_flatten(init_y), 'x' : tensor_flatten(preds[:,0]), 'y': tensor_flatten(preds[:,1])}, ["time", "initial_x", "initial_y", "x", "y"])



def predicting_lorenz_system():
    """
        predicting_lorenz_system:
            This will be used for predicting the lorenz system
    """
    filename = "data/train/lorenz.csv"
    inputs = inputs_to_tensor(filename, [0,4,5,6])
    targets = inputs_to_tensor(filename, [1, 2, 3])

    preds, _ = predictions(num_inputs=4, num_classes=3, learning_rate=0.000000005, batch_size=15, num_epochs=100, inputs=inputs, targets=targets, train=True, path="lorenz.pth")
    time = pd.read_csv(filename, usecols=[0])

    init_x = pd.read_csv(filename, usecols=[4])
    init_y = pd.read_csv(filename, usecols=[5])
    init_z = pd.read_csv(filename, usecols=[6])

    save("data/preds/train/lorenz.csv", {'time' : df_flatten(time), 'initial_x' : df_flatten(init_x), 'initial_y' : df_flatten(init_y), 'initial_z' : df_flatten(init_z), 'x' : tensor_flatten(preds[:,0]), 'y' : tensor_flatten(preds[:,1]), 'z' : tensor_flatten(preds[:,2]) }, ["time", "initial_x", "initial_y", "initial_z", "x", "y", "z"])


def predicting_laub_loomis():
    """
        laub_loomis:
            This will be used for predicting the lorenz system
    """
    filename = "data/train/laub.csv"
    columns = [i for i in range (8,15)]
    columns = columns.append(0)
    inputs = inputs_to_tensor(filename, [0,8,9,10,11,12,13,14])
    targets = inputs_to_tensor(filename, [1,2,3,4,5,6,7])
 

    EPOCHS = 10
    BATCH = 500

    preds, _ = predictions(num_inputs=8, num_classes=7, learning_rate=0.0001, batch_size=BATCH, num_epochs=EPOCHS, inputs=inputs, targets=targets, train=True, path="loomis.pth")

    time = pd.read_csv(filename, usecols=[0])
    init_x = pd.read_csv(filename, usecols=[8])
    init_y = pd.read_csv(filename, usecols=[9])
    init_z = pd.read_csv(filename, usecols=[10])
    init_w = pd.read_csv(filename, usecols=[11])
    init_p = pd.read_csv(filename, usecols=[12])
    init_q = pd.read_csv(filename, usecols=[13])
    init_m = pd.read_csv(filename, usecols=[14])

    save("data/preds/train/loomis.csv", {'time' : df_flatten(time), 
        'initial_x' : df_flatten(init_x), 
        'initial_y' : df_flatten(init_y), 
        'initial_z' : df_flatten(init_z), 
        'initial_w' : df_flatten(init_w),
        'initial_p' : df_flatten(init_p), 
        'initial_q' : df_flatten(init_q),
        'initial_m' : df_flatten(init_m), 
        'x' : tensor_flatten(preds[:,0]),
        'y' : tensor_flatten(preds[:,1]),
        'z' : tensor_flatten(preds[:,2]),
        'w' : tensor_flatten(preds[:,3]),
        'p' : tensor_flatten(preds[:,4]),
        'q' : tensor_flatten(preds[:,5]),
        'm' : tensor_flatten(preds[:,6]),
         }, ["time", "initial_x", "initial_y", "initial_z", "initial_w", "initial_p", "initial_q", "initial_m", "x", "y", "z", "w", "p", "q", "m"])


if __name__== "__main__":
    predicting_laub_loomis()
