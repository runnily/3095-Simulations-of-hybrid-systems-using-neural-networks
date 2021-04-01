from utilities import LossUtilities, np, sys
from torch import Tensor
from itertools import product
sys.path.insert(1, '../')
from simulations import van_der_pol_oscillator, laub_loomis, lorenz_system

"""
    Arthur: Adanna Obibuaku
    Purpose: This will module is used for define the abstract class. Each class defined correspond to a
             Model. Within the class it define their default inputs and simulation method, for defining
             a simulation when needed.
    Date:   29/03/21
"""

class NewtonsLoss(LossUtilities):
  
    def default_model_inputs(self):
        default_lr = 0.0001
        default_batch_size = 50
        default_time_step = 1
        default_num_epoches = 100
        filename = "../data/train/newtons_cooling_law.csv"
        inputs = self.inputs_to_tensor(filename, [0,1])
        targets = self.inputs_to_tensor(filename, [2])
        number_inputs = 2
        number_classes = 1
        
        return default_lr, default_batch_size, default_time_step, default_num_epoches, number_inputs, number_classes, inputs, targets

class VanDerPol(LossUtilities):

    def simulations(self, delta):
        df_simulations = van_der_pol_oscillator(delta, False)
        inputs = df_simulations[['time','initial_x','initial_y']].to_numpy(dtype='float32')
        outputs = df_simulations[['x','y']].to_numpy(dtype='float32')
        return Tensor(inputs), Tensor(outputs)

    def default_model_inputs(self):
        default_lr = 0.0005
        default_batch_size = 15
        default_time_step = 0.001
        default_num_epoches = 20
        filename = "../data/train/van.csv"
        inputs = self.inputs_to_tensor(filename, [0,3,4])
        targets = self.inputs_to_tensor(filename, [1,2])
        number_inputs = 3
        number_classes = 2
        
        return default_lr, default_batch_size, default_time_step, default_num_epoches, number_inputs, number_classes, inputs, targets

class lorenz(LossUtilities):
  
    def simulations(self, delta):
        df_simulations = van_der_pol_oscillator(delta, False)
        inputs = df_simulations[['time','initial_x','initial_y']].to_numpy(dtype='float32')
        outputs = df_simulations[['x','y']].to_numpy(dtype='float32')
        return Tensor(inputs), Tensor(outputs)

    def default_model_inputs(self):
        default_lr = 0.0001
        default_batch_size = 500
        default_time_step = 0.01
        default_num_epoches = 10
        filename = "../data/train/lorenz.csv"
        inputs = self.inputs_to_tensor(filename, [0,4,5,6])
        targets = self.inputs_to_tensor(filename, [1, 2, 3])
        number_inputs = 4
        number_classes = 3
        return default_lr, default_batch_size, default_time_step, default_num_epoches, number_inputs, number_classes, inputs, targets

class laub(LossUtilities):

    def simulations(self, delta):
        df_simulations = lorenz_system(delta, False)
        inputs = df_simulations[['time','initial_x','initial_y',
        'initial_z']].to_numpy(dtype='float32')
        outputs = df_simulations[['x','y','z']].to_numpy(dtype='float32')
        return Tensor(inputs), Tensor(outputs)

    def default_model_inputs(self):
        default_lr = 0.0001
        default_batch_size = 500
        default_time_step = 0.01
        default_num_epoches = 10
        filename = "../data/train/laub.csv"
        inputs = self.inputs_to_tensor(filename, [0,8,9,10,11,12,13,14])
        targets = self.inputs_to_tensor(filename, [1,2,3,4,5,6,7])
        number_inputs = 8
        number_classes = 7  
        return default_lr, default_batch_size, default_time_step, default_num_epoches, number_inputs, number_classes, inputs, targets