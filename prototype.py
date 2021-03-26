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

    preds = predictions(num_inputs=1, num_classes=1, learning_rate=0.000001, batch_size=301000, num_epochs=200, inputs=inputs, targets=targets, train=True, path="simple_model_x0.pth" )
    save("data/preds/train/simple_model_x0.csv",  {'x' : inputs.numpy().flatten(), 'y' : preds}, ["x", "y"])

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

    preds = predictions(num_inputs=1, num_classes=1, learning_rate=0.00001, batch_size=50, num_epochs=2000, inputs=inputs, targets=targets, train=True, path="simple_model_x1.pth" )
    save("data/preds/train/simple_model_x1.csv",  {'x' : inputs.numpy().flatten(), 'y' : preds}, ["x", "y"])

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

    preds = predictions(num_inputs=1, num_classes=1, learning_rate=0.00001, batch_size=50, num_epochs=2000, inputs=inputs, targets=targets, train=True, path="simple_model_y2.pth" )
    save("data/preds/train/simple_model_y2.csv",  {'x' : inputs.numpy().flatten(), 'y' : preds}, ["x", "y"])

def predicting_simple_model_x1y2():
    """
        This is the modal correspond to the 2 behvious in the second model, I specifed in the paper. Both equations
        work together. 
    """

    # Get inputs and targets from file 
    inputs = inputs_to_tensor("data/train/simple_model_x1.csv", [0])
    targets = inputs_to_tensor("data/train/simple_model_y2.csv", [1])

    preds = predictions(num_inputs=1, num_classes=1, learning_rate=0.000001, batch_size=50, num_epochs=5000, inputs=inputs, targets=targets, train=True, path="simple_model_x1y2.pth" )
    save("data/preds/train/simple_model_x1y2.csv",  {'x' : inputs.numpy().flatten(), 'y' : preds}, ["x", "y"])

def predicting_newtons_cooling_law(filename, train):
    """
        predicting_cooling:
            This will be used uses the neural network to simulate the dynamics of newtons cooling laws.
    """
    inputs = inputs_to_tensor(filename, [0,1])
    targets = inputs_to_tensor(filename, [2])

    if train:
        preds = predictions(num_inputs=2, num_classes=1, learning_rate=0.0001, batch_size=50, num_epochs=1000, inputs=inputs, targets=targets, train=True, path="newtons_cooling_law.pth")
        savefile = "data/preds/train/newtons_cooling_law.csv"
    else:
        preds = predictions(num_inputs=2, num_classes=1, learning_rate=0.00001, batch_size=50, num_epochs=600, inputs=inputs, targets=targets, train=False, path="newtons_cooling_law.pth")
        savefile = "data/preds/test/newtons_cooling_law.csv"

    init_temp = pd.read_csv(filename, usecols=[0])
    time_temp = pd.read_csv(filename, usecols=[1])

    
    save(savefile, {'initial_temp' : init_temp.values.flatten(), 'time' : time_temp.values.flatten(), 'temp' : preds}, ["initial_temp", "time", "temp"])
    return savefile

def predicting_van_der_pol():
    """
        predicting_van_der_pol:
            This will be used uses the neural network to simulate the dynamics of the van
            der pol method
    """
    filename = "data/train/van.csv"
    inputs = inputs_to_tensor(filename, [0,3,4])
    targets_x = inputs_to_tensor(filename, [1])
    targets_y = inputs_to_tensor(filename, [2])

    # 1000
    # when it was 10, it was 5
    preds_x = predictions(num_inputs=3, num_classes=1, learning_rate=0.0005, batch_size=15, num_epochs=25, inputs=inputs, targets=targets_x, train=True, path="van_der_pol/vans_x.pth")
    preds_y = predictions(num_inputs=3, num_classes=1, learning_rate=0.0005, batch_size=15, num_epochs=25, inputs=inputs, targets=targets_y, train=True, path="van_der_pol/vans_y.pth")

    time = pd.read_csv(filename, usecols=[0])
    init_x = pd.read_csv(filename, usecols=[3])
    init_y = pd.read_csv(filename, usecols=[4])

    save("data/preds/train/van.csv", {'time' : time.values.flatten(), 'initial_x' : init_x.values.flatten(), 'initial_y' : init_y.values.flatten(), 'x' : preds_x, 'y' : preds_y }, ["time", "initial_x", "initial_y", "x", "y"])


def predicting_lorenz_system():
    """
        predicting_lorenz_system:
            This will be used for predicting the lorenz system
    """
    filename = "data/train/lorenz.csv"
    inputs = inputs_to_tensor(filename, [0,4,5,6])
    targets_x = inputs_to_tensor(filename, [1])
    targets_y = inputs_to_tensor(filename, [2])
    targets_z = inputs_to_tensor(filename, [3])

    preds_x = predictions(num_inputs=4, num_classes=1, learning_rate=0.1, batch_size=5287, num_epochs=5, inputs=inputs, targets=targets_x, train=True, path="lorenz/lorenz_x.pth")
    preds_y = predictions(num_inputs=4, num_classes=1, learning_rate=0.1, batch_size=5287, num_epochs=5, inputs=inputs, targets=targets_y, train=True, path="lorenz/lorenz_y.pth")
    preds_z = predictions(num_inputs=4, num_classes=1, learning_rate=0.1, batch_size=5287, num_epochs=5, inputs=inputs, targets=targets_z, train=True, path="lorenz/lorenz_z.pth")

    time = pd.read_csv(filename, usecols=[0])
    init_x = pd.read_csv(filename, usecols=[4])
    init_y = pd.read_csv(filename, usecols=[5])
    init_z = pd.read_csv(filename, usecols=[6])

    save("data/preds/train/lorenz.csv", {'time' : time.values.flatten(), 'initial_x' : init_x.values.flatten(), 'initial_y' : init_y.values.flatten(), 'initial_z' : init_z.values.flatten(), 'x' : preds_x, 'y' : preds_y, 'z' : preds_z }, ["time", "initial_x", "initial_y", "initial_z", "x", "y", "z"])


def predicting_laub_loomis():
    """
        laub_loomis:
            This will be used for predicting the lorenz system
    """
    filename = "data/train/laub.csv"
    columns = [i for i in range (8,15)]
    columns = columns.append(0)
    inputs = inputs_to_tensor(filename, [0,8,9,10,11,12,13,14])
    targets_x = inputs_to_tensor(filename, [1])
    targets_y = inputs_to_tensor(filename, [2])
    targets_z = inputs_to_tensor(filename, [3])
    targets_w = inputs_to_tensor(filename, [4])
    targets_p = inputs_to_tensor(filename, [5])
    targets_q = inputs_to_tensor(filename, [6])
    targets_m = inputs_to_tensor(filename, [7])

    EPOCHS = 10
    BATCH = 500

    preds_x = predictions(num_inputs=8, num_classes=1, learning_rate=0.0001, batch_size=BATCH, num_epochs=EPOCHS, inputs=inputs, targets=targets_x, train=True, path="loomis/loomis_x.pth")
    preds_y = predictions(num_inputs=8, num_classes=1, learning_rate=0.0001, batch_size=BATCH, num_epochs=EPOCHS, inputs=inputs, targets=targets_y, train=True, path="loomis/loomis_y.pth")
    preds_z = predictions(num_inputs=8, num_classes=1, learning_rate=0.0001, batch_size=BATCH, num_epochs=EPOCHS, inputs=inputs, targets=targets_z, train=True, path="loomis/loomis_z.pth")
    preds_w = predictions(num_inputs=8, num_classes=1, learning_rate=0.0001, batch_size=BATCH, num_epochs=EPOCHS, inputs=inputs, targets=targets_w, train=True, path="loomis/loomis_w.pth")
    preds_p = predictions(num_inputs=8, num_classes=1, learning_rate=0.0001, batch_size=BATCH, num_epochs=EPOCHS, inputs=inputs, targets=targets_p, train=True, path="loomis/loomis_p.pth")
    preds_q = predictions(num_inputs=8, num_classes=1, learning_rate=0.0001, batch_size=BATCH, num_epochs=EPOCHS, inputs=inputs, targets=targets_q, train=True, path="loomis/loomis_q.pth")
    preds_m = predictions(num_inputs=8, num_classes=1, learning_rate=0.0001, batch_size=BATCH, num_epochs=EPOCHS, inputs=inputs, targets=targets_m, train=True, path="loomis/loomis_m.pth")

    time = pd.read_csv(filename, usecols=[0])
    init_x = pd.read_csv(filename, usecols=[8])
    init_y = pd.read_csv(filename, usecols=[9])
    init_z = pd.read_csv(filename, usecols=[10])
    init_w = pd.read_csv(filename, usecols=[11])
    init_p = pd.read_csv(filename, usecols=[12])
    init_q = pd.read_csv(filename, usecols=[13])
    init_m = pd.read_csv(filename, usecols=[14])

    save("data/preds/train/loomis.csv", {'time' : time.values.flatten(), 
        'initial_x' : init_x.values.flatten(), 
        'initial_y' : init_y.values.flatten(), 
        'initial_z' : init_z.values.flatten(), 
        'initial_w' : init_w.values.flatten(), 
        'initial_p' : init_p.values.flatten(), 
        'initial_q' : init_q.values.flatten(),
        'initial_m' : init_m.values.flatten(), 
        'x' : preds_x,
        'y' : preds_y, 
        'z' : preds_z,
        'w' : preds_w,
        'p' : preds_p, 
        'q' : preds_q,
        'm' : preds_m,
         }, ["time", "initial_x", "initial_y", "initial_z", "initial_w", "initial_p", "initial_q", "initial_m", "x", "y", "z", "w", "p", "q", "m"])


if __name__== "__main__":
   predicting_lorenz_system()
   predicting_laub_loomis()
   predicting_newtons_cooling_law("data/train/newtons_cooling_law.csv", True)