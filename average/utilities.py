"""
    Arthur: Adanna Obibuaku
    Purpose: This module is acts as a helper for collecting 30 loss data item of the neural network, depedant on the parameters.
             This defines an abstract class which will be used to collect 30 loss data items, will specifiying certain parameters
             that should be that correspond to the 30 points.
    Date:   29/03/21
"""
import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.insert(1, '../')
from prototype import predictions, inputs_to_tensor
from torch.utils.data import TensorDataset, DataLoader # For mini batches
from abc import ABC, abstractmethod
import multiprocessing
from joblib import Parallel, delayed
from prototype_nn import CustomeModel

class LossUtilities():
    """
        lossUtilities:
            Acts as a helper class for all the other modules. This would be used for gather data for plotting box plots
            
    """
    @abstractmethod
    def simulations(self, delta):
        """
            simulation: This will perform a simulation of the model
            Args:
                delta (float): This denotes the time step that is used
            Returns:
                (class <DataFrame>): This would will produce the dataframe of the simulations.
        """
        pass

    @abstractmethod
    def default_model_inputs(self):
        """
            default_model_input: An abstract class which will be used for return their default inputs for the models and 
                                 corresponding neural network model.
            Args:
                None
            Returns:
                (int) : Denotes the default learning rate
                (int) : Denotes the default batch size
                (float) : Denotes the default time step
                (int) : Denotes the number of epoches being used
                (int) : Denotes the number of inputs being used.
                (class <Tensor>) : Denotes the inputs being put into the neural network
                (class <Tensor>) : Densotes the targets of the neural network

        """
        pass
    
    def inputs_to_tensor(self, filename, columns):
        """
            inputs_to_tensor:
                This would turn a dataframe into a tensor
            Args:
                filename (string): The filename which we want to read from
                columns (class <list>): The columns which we want to use in the file
            Returns:
                <class <Tensor>): Returns a tensor of the dataframe
        """
        return inputs_to_tensor(filename, columns)
    
    def loss(self, len_parameters, **list_of_parameters):
        """
            loss: For each parameter that is defined in list of parameter it will produce 30 loss items that correspond to what was
                  specified in the list of parameter.
            
            Args:
                list_of_paramters (class <dict>) : Is a dictionary with a list of paramters
                len_parameter: This is the length of the parameters
            Returns:
                (class <list>) : An array of the loss
                (<DataFrame>) : The dataframe of arrays
        """
        # checks if the len of each array in len_parameters is the same
        for key, values in list_of_parameters.items():
            if len(list(values)) != len_parameters:
                raise ValueError("length of {} should be the same as {}".format(key, len_parameters))
                
        # get the defaults values
        
        default_lr, default_batch_size, default_time_step, default_num_epoches, number_inputs, number_classes, inputs, targets = self.default_model_inputs()
        # sets the list of parameters to None or a list of values to go through
        lr = list_of_parameters.get("lr", None)
        batch_size = list_of_parameters.get("batch_size", None)
        time_step = list_of_parameters.get("time_step", None)
        num_epoches = list_of_parameters.get("num_epoches", None)

        loss_dict = {} # This hold a dictionary containing the parameters and their corresponding loss
        for i in range(len_parameters):
            # These are setting the parameters to either default parameter or a 
            # specific parameter defined within a list
            learning_rate = default_lr if lr == None else lr[i]
            batch_size_1 = default_batch_size if batch_size == None else batch_size[i]
            if time_step != None:
                time_step_1 = time_step[i]
                # This will runs a simulations with the defined time step (delta)
                # Then use the ressasign inputs and targets to now correspond to the new delta
                # This would then later be used for caculations
                inputs, targets = self.simulations(time_step_1) 
            else:
                time_step_1 = default_batch_size
                
            num_epoches_1 = default_num_epoches if num_epoches == None else num_epoches[i]
            
            # Runs in parrell multiple times.
            loss = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(predictions)(number_inputs, number_classes, learning_rate, batch_size_1, num_epoches_1, inputs, targets, True, None, None) for _ in range(30))

            # [x[1] for x in loss] denotes the second value within the tuples of a list
            loss_dict["lr = {}, bs = {}, ts = {}, epoch = {}".format(learning_rate, batch_size_1,time_step_1,num_epoches_1)] = [x[1] for x in loss]

        return loss_dict, pd.DataFrame(data=loss_dict)

    def loss_modeling(self, layer_list_of_layers):
        """
            loss_modeling:
                This is uses different types of layers defined in layer_list_of_layers to check how the different number of parameters and 
                layers affect the loss
            Args: 
                layer_list_of_layers: 
                    Format example: [ [{"in_features" : x, "out_features" : y}, 
                                        {"in_features" : x_1, "out_features" : y1},
                                        {"in_features" : x_2, "out_features" : y_2},]
                                        [{"in_features" : x_3, "out_features" : y_3}] ]
                    Explained: We define an array x. Within array x we have a set amount of n arrays denoted as Y
                               Y = y_1, y_2, y_3, ... y_n. Within each Y array is a dictionary.
                    **Must be in the format denoted above to work**
            
        """
        loss_dict = {}
        for layer_list in layer_list_of_layers:
            loss = self.loss_model(layer_list)
            for x in layer_list:
                print(x.items())
            loss_dict[key] = loss
        return loss_dict, pd.DataFrame(data=loss_dict)

    def loss_model(self, layer_list):
        """
            loss_model: Makes use of the class CustomeModel, to allow user to create a neural network model
                        based on the layers defined within the layer_list. 
        """

        default_lr, default_batch_size, default_time_step, default_num_epoches, number_inputs, number_classes, inputs, targets = self.default_model_inputs()
        model = CustomeModel(number_inputs, number_classes, default_lr, layer_list)
        loss = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(predictions(number_inputs, number_classes, default_lr, default_batch_size, default_num_epoches, inputs, targets, True, path = None, model = model)) for _ in range(30))
   
        # [x[1] for x in loss] denotes the second item within the tuples of a list
        loss = [x[1] for x in loss]
        return loss
        