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

class LossUtilities():
    """
        lossUtilities:
            Acts as a helper class for all the other modules. This would be used for gather data for plotting box plots
            
    """
    @abstractmethod
    def simulations(self, delta):
        pass

    @abstractmethod
    def default_model_inputs(self):
        pass
    
    def inputs_to_tensor(self, filename, columns):
        return inputs_to_tensor(filename, columns)
    
    def loss(self, len_parameters, **list_of_parameters):
        """
            loss: This would be using for gathering data at least 30 times. To then be used for predictions.
            
            Args:
                list_of_paramters <dict> : Is a dictionary with a list of paramters
                len_parameter: This is the length of the parameters
            Returns:
                <array> : An array of the loss
                <DataFrame> : The dataframe of arrays
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
            loss = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(predictions)(number_inputs, number_classes, learning_rate, batch_size_1, num_epoches_1, inputs, targets, True, None) for _ in range(30))

            loss_dict["lr = {}, bs = {}, ts = {}, epoch = {}".format(learning_rate, batch_size_1,time_step_1,num_epoches_1)] = [loss[0][1]]

        return loss_dict, pd.DataFrame(data=loss_dict)