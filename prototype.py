"""
    Arthur: Adanna Obibuaku
    Purpose: This will be used for building neural network models. These neural network models
             will be used to for the predictions of a simulation given an input. These models will be
             tested later to see their accuracy
    Date: 09/02/21
"""
import prototype_nn as NN
from prototype_nn import save, predictions, inputs_to_tensor, df_flatten, tensor_flatten
from sklearn import preprocessing as pre
from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader # For mini batches
import pandas as pd
MAIN_PATH = "data/state/"

def predicting_newtons_cooling_law():
    """
        predicting_cooling:
            This will be used uses the neural network to simulate the dynamics of newtons cooling laws.
    """
    
    filename = "data/train/newtons_cooling_law.csv"
    inputs = inputs_to_tensor(filename, [0,2])
    targets = inputs_to_tensor(filename, [1])
    preds, _= predictions(num_inputs=2, num_classes=1, learning_rate=0.0001, batch_size=32, num_epochs=30, inputs=inputs, targets=targets, train=True, path= MAIN_PATH + "newtons_cooling_law.pth")

    init_temp = pd.read_csv(filename, usecols=[2])
    time = pd.read_csv(filename, usecols=[0])
    
    save("data/preds/train/newtons_cooling_law.csv", {'initial_temp' : df_flatten(init_temp), 'time' : df_flatten(time), 'temp' : tensor_flatten(preds)}, ["initial_temp", "time", "temp"])

def predicting_van_der_pol():
    """
        predicting_van_der_pol:
            This will be used uses the neural network to simulate the dynamics of the van
            der pol method
    """
    filename = "data/train/van.csv"
    inputs = inputs_to_tensor(filename, [0,3,4])
    targets = inputs_to_tensor(filename, [1, 2])

    preds, _ = predictions(num_inputs=3, num_classes=2, learning_rate=0.00001, batch_size=32, num_epochs=20, inputs=inputs, targets=targets, train=True, path= MAIN_PATH + "van.pth")
    
    time = pd.read_csv(filename, usecols=[0])
    init_x = pd.read_csv(filename, usecols=[3])
    init_y = pd.read_csv(filename, usecols=[4])
   
    save("data/preds/train/van.csv", {'time' : df_flatten(time), 'initial_x' : df_flatten(init_x), 'initial_y' : df_flatten(init_y), 'x' : tensor_flatten(preds[:,0]), 'y': tensor_flatten(preds[:,1])}, ["time", "initial_x", "initial_y", "x", "y"])


def predicting_laub_loomis():
    """
        laub_loomis:
            This will be used for predicting the lorenz system
    """
    filename = "data/train/laub.csv"
    inputs = inputs_to_tensor(filename, [0,8,9,10,11,12,13,14])
    targets = inputs_to_tensor(filename, [1,2,3,4,5,6,7])
 

    EPOCHS = 5
    BATCH = 32

    preds, _ = predictions(num_inputs=8, num_classes=7, learning_rate=0.00005, batch_size=BATCH, num_epochs=EPOCHS, inputs=inputs, targets=targets, train=True, path= MAIN_PATH + "loomis.pth")

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


def predicting_biological_model():
    """
        biological_model:
            This will be used for predicting the lorenz system
    """
    filename = "data/train/biological_model.csv"
    inputs = inputs_to_tensor(filename, [0,10,11,12,13,14,15,16,17,18])
    targets = inputs_to_tensor(filename, [1,2,3,4,5,6,7,8,9])
 

    EPOCHS = 5
    BATCH = 32

    preds, _ = predictions(num_inputs=10, num_classes=9, learning_rate=0.00005, batch_size=BATCH, num_epochs=EPOCHS, inputs=inputs, targets=targets, train=True, path= MAIN_PATH + "biological.pth")

    time = pd.read_csv(filename, usecols=[0])
    init_x1 = pd.read_csv(filename, usecols=[10])
    init_x2 = pd.read_csv(filename, usecols=[11])
    init_x3 = pd.read_csv(filename, usecols=[12])
    init_x4 = pd.read_csv(filename, usecols=[13])
    init_x5 = pd.read_csv(filename, usecols=[14])
    init_x6 = pd.read_csv(filename, usecols=[15])
    init_x7 = pd.read_csv(filename, usecols=[16])
    init_x8 = pd.read_csv(filename, usecols=[17])
    init_x9 = pd.read_csv(filename, usecols=[18])

    save("data/preds/train/biological_model.csv", {'time' : df_flatten(time), 
        'initial_x1' : df_flatten(init_x1), 
        'initial_x2' : df_flatten(init_x2), 
        'initial_x3' : df_flatten(init_x3), 
        'initial_x4' : df_flatten(init_x4),
        'initial_x5' : df_flatten(init_x5),
        'initial_x6' : df_flatten(init_x6), 
        'initial_x7' : df_flatten(init_x7),
        'initial_x8' : df_flatten(init_x8), 
        'initial_x9' : df_flatten(init_x9), 
        'x1' : tensor_flatten(preds[:,0]),
        'x2' : tensor_flatten(preds[:,1]),
        'x3' : tensor_flatten(preds[:,2]),
        'x4' : tensor_flatten(preds[:,3]),
        'x5' : tensor_flatten(preds[:,4]),
        'x6' : tensor_flatten(preds[:,5]),
        'x7' : tensor_flatten(preds[:,6]),
        'x8' : tensor_flatten(preds[:,7]),
        'x9' : tensor_flatten(preds[:,8]),
         }, ["time", "initial_x1", "initial_x2", "initial_x3", "initial_x4", "initial_x5", "initial_x6", "initial_x7", "initial_x8", "initial_x9", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"])

def predicting_bouncing_ball():
    """
        predicting_bouncing_ball:
            This will be used uses the neural network to simulate the dynamics of a bouncing ball
    """
    filename = "data/train/bouncing.csv"

    inputs = inputs_to_tensor(filename, [0,1,2,4])
    targets = inputs_to_tensor(filename, [3, 4])
    
    preds, _ = predictions(num_inputs = 4, num_classes=2, learning_rate=0.0001, batch_size=32, num_epochs=20, inputs=inputs, targets=targets, train=True, path= MAIN_PATH + "bouncing.pth")
    
    time = pd.read_csv(filename, usecols=[0])
    init_p = pd.read_csv(filename, usecols=[1])
    init_v = pd.read_csv(filename, usecols=[2])
   
    #save("data/preds/train/bouncing.csv", {'time' : df_flatten(time), 'initial_position' : df_flatten(init_p), 'initial_velocity' : df_flatten(init_v), 'position' : tensor_flatten(preds[:,0])}, ["time", "initial_position", "initial_velocity", "position", ])

    save("data/preds/train/bouncing.csv", {'time' : df_flatten(time), 'initial_position' : df_flatten(init_p), 'initial_velocity' : df_flatten(init_v), 'position' : tensor_flatten(preds[:,0]), 'velocity': tensor_flatten(preds[:,1])}, ["time", "initial_position", "initial_velocity", "position", "velocity"])


def predicting_spiking_neurons():
    """
        predicting_bouncing_ball:
            This will be used uses the neural network to simulate the dynamics of a bouncing ball
    """
    filename = "data/train/spiking_neurons.csv"

    inputs = inputs_to_tensor(filename, [0,1,2,4])
    targets = inputs_to_tensor(filename, [3, 4])
    
    preds, _ = predictions(num_inputs = 4, num_classes=2, learning_rate=0.0001, batch_size=64, num_epochs=100, inputs=inputs, targets=targets, train=True, path= MAIN_PATH + "spiking_neurons.pth")
    
    time = pd.read_csv(filename, usecols=[0])
    init_u = pd.read_csv(filename, usecols=[1])
    init_v = pd.read_csv(filename, usecols=[2])
   
    #save("data/preds/train/bouncing.csv", {'time' : df_flatten(time), 'initial_position' : df_flatten(init_p), 'initial_velocity' : df_flatten(init_v), 'position' : tensor_flatten(preds[:,0])}, ["time", "initial_position", "initial_velocity", "position", ])

    save("data/preds/train/spiking_neurons.csv", {'time' : df_flatten(time), 'initial_u' : df_flatten(init_u), 'initial_v' : df_flatten(init_v), 'u' : tensor_flatten(preds[:,0]), 'v': tensor_flatten(preds[:,1])}, ["time", "initial_u", "initial_v", "u", "v"])



if __name__== "__main__":
 print("... calculating predictions ...")
 predicting_spiking_neurons()