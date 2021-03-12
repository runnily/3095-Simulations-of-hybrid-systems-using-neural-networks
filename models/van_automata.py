"""
    Arthur: Adanna Obibuaku
    Purpose: The purpose of this is too build the van der pool model
"""
from automata import Automata
import decimal

class AutomataSys(Automata):
    def __init__(self, current, states, guards, x_0 = None, x_env = None):
        super(AutomataSys, self).__init__(current, states, guards, x_0 = None, x_env = None)

        # This will but all behaviours in a list
        self.rate_of_change_id = ['dx/dtime', 'dy/dtime', 'dz/dtime', 'dw/dtime']
        self.rate_of_change = {}

        # Assignes each behaviours to an corresponding id
        for i, behaviour in enumerate(self.current.behaviour):
            self.rate_of_change[self.rate_of_change_id[i]] =  behaviour

    def transitions(self, x):
        """
            TODO
        """
        pass

    def run(self, initial_var, delta, num_simulations):
        text = ''
        time = 0

        # Will hold all variables in the contious systems and the current value
        values = {}

        if len(initial_var) != len(self.rate_of_change):
            raise IndentationError("Length of inital_var must be the same as length of rate of change")
        else:
            # Goes through the each equations in the systems of differential equations,
            # To defind a variable name and the current state value which is 
            # initially the initial var provided by initial_var
            for i, (key, value) in enumerate(self.rate_of_change.items()): 
                # values is used to store the variables within the behaviour
                # the key[1] is used to get the second character within the string key in rate of change items
                # This would be then used as a key for the values
                # This is first stored as initial_var
                values[key[1]] = initial_var[i] 
        
        for _ in range(num_simulations):
            text += "{time},".format(time=round(time,1))
            update_values = {}

            for key, value in values.items():
                # For printing every value in the data
                text += "{key},".format(key=value) 
            
            list_keys = self.rate_of_change.keys()
            for key, behaviour in self.rate_of_change.items():
                # Gets the arguments of the behaviour as a function
                func_arguments = behaviour.__code__.co_varnames
                # holds the arguments to be passed into the behaviour
                func_inputs = {}
                # This would go through each arguments in arguments
                # Find the corresponding values in values
                # And add it the directory inputs
                # which is then passed to the behaviour


                for argument in func_arguments:
                    func_inputs[argument] = values[argument]
                
                rate_change = behaviour(**func_inputs)
                if (list(list_keys)[-1] != key):
                    text += "{rate_change},".format(rate_change=rate_change) 
                else:
                    text += "{rate_change}".format(rate_change=rate_change) 

                update_values[key[1]] = values[key[1]]+rate_change*delta

            values = update_values
            
            text += "\n"
            time += delta
        print(text)
        fieldnames = ['t'] + list(values.keys()) + list(self.rate_of_change.keys())
        self.save(text, fieldnames, '../data/train/van.csv')





