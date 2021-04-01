"""
Arthur:
    Adanna Obibuaku
Purpose:    
    This module implements a hybrid automata class. This is a standard class
    for reperesenting all models of hybrid automata
"""

import csv 

class Automata:
    """
    Automata: This class can be used for reperesenting your automata

    Attributes:
        lab (<class 'dict'>): This is used for reperesent a dictionary where the guards (edges) are the keys pointing to 
                             a specfific state it jumps to.
    """

    def __init__(self, current, states, guards, x_0 = None, x_env = None):
        """
        __init__: This is used to inilise the hybrid automaton object

        Args:
            current (<class 'state.State'>): This reperesent the current state of the hybrid automaton
            states (<class 'list'>): This reperesents a list of states used within the hybrid automaton
            guards (<class 'list'>): This reperesent a list of guards used with the hybrid automaton
            x_0 (int): This reperesents the initial x 
            x_env (int): This repersents the enviroment x if included.

        """
        self.states = states
        self.guards = guards
        self.current = current

        self.x_0 = x_0
        self.x_env = x_env

        self.lab = {}
        for i in range(len(guards)):
            self.lab[guards[i]] = states[i]
     
    
    def transitions(self, x):
        """
        transitions: This acts as a discerete event, where a guard condition is met we jump to the assiocated 
                     state the guard is pointing to 
        Args:
            x (float): Reperesents the dynamic input
        """
        for guard in self.guards: 
            if guard(x) and self.lab[guard].invariant(x):
                    self.current = self.lab[guard]

    def save(self, text, fieldnames, filename):
        with open(filename, "a+") as file:
            if file.tell() == 0:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
            file.write(text)

    def run(self, y0, delta, num_simulations, filename, x=None):
        """
        run: The evolution of the state of the hybrid system over time. This run is achieved
             by performing the euler method

        Args:
            y0 (float): The inital state of the system at the beginning.
            delta (float): The increments we should change by
            noOfSimulations (int): How many numertical simulations we want
        """
        y = y0
        if x == None:
            x = 0
        text = ""
        fieldnames = None

        for _ in range(num_simulations):
            dydx = self.current.behaviour(y) # Get the change rate of change
            self.transitions(y) # To change the state if needed
            dydx_string = "dydx"
            if (self.x_0 == None) and (self.x_env == None):
                text += '{x},{y},{dydx},"{state}"\n'.format(x=x, y=y,dydx=dydx, state=self.current.name)
                fieldnames = ["x", "y", dydx_string, "state"]
            elif (self.x_0 != None) and (self.x_env == None):
                text += '{x_0},{x},{y},{dydx},"{state}"\n'.format(x_0 = self.x_0, x=x, y=y,dydx=dydx, state=self.current.name)
                fieldnames = ["x_0", "x", "y", dydx_string, "state"]
            elif (self.x_0 == None) and (self.x_env != None):
                text += '{x_env},{x},{y},{dydx},"{state}"\n'.format(x_env = self.x_env, x=x, y=y,dydx=dydx, state=self.current.name)
                fieldnames = ["x_env", "x", "y", dydx_string, "state"]
            else:
                text += '{x_0},{x_env},{x},{y},{dydx},"{state}"\n'.format(x_0 = self.x_0, x_env = self.x_env, x=x, y=y,dydx=dydx, state=self.current.name)
                fieldnames = ["x_0", "x_env", "x", "y", dydx_string, "state"]

            y += dydx*delta # update the change
            x += delta # delta

        self.save(text, fieldnames, filename)
            
            
class AutomataSys(Automata):
    def __init__(self, current, states, guards, x_0 = None, x_env = None):
        super(AutomataSys, self).__init__(current, states, guards, x_0 = None, x_env = None)

        # This will but all behaviours in a list
        self.rate_of_change_id = ['dx/dtime', 'dy/dtime', 'dz/dtime', 'dw/dtime', 'dp/dtime', 'dq/time','dm/time']
        self.rate_of_change = {}

        # Assignes each behaviours to an corresponding id
        for i, behaviour in enumerate(self.current.behaviour):
            self.rate_of_change[self.rate_of_change_id[i]] =  behaviour

    def transitions(self, x):
        """
            TODO
        """
        pass

    def variables_of_systems(self, initial_var):
        """
            variables_of_systems:
                This is used to generate a dictionary of all variables within the continous system
                to correspond to its initial value, which is stored in initial_var
            Args:
                initial_var (<class 'list'>): This is a list of all variables within the system
            Returns:
                (<class 'dict'>) : A dictionary of all variables keys corresponding to the current value
                (<class 'dict'>) : A dictionary of all variables keys corresponding to the initial value (should remain the same)
        """
        values = {}
        initial_var_dict = {}
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
                initial_var_dict["initial_"+key[1]] = initial_var[i] 
        return values, initial_var_dict

    def run(self, initial_var, delta, num_simulations, filename = None):
        """
            run:
                This is to run the euler method on a set of behaviours in an hybrid automata
                which is defined with a systems of behaviours.
            Args:
                initial_var (<class 'list'>): This is a list of all variables within the system
                delta (int): This is a the time step to take. This initial variables should
                correspond to the differinal equations.
                num_simulations: This is the number of simulations that should be made
        """
        text = ''
        time = 0

        # Will hold all variables in the contious systems and the current value
        values, initial_var_dict = self.variables_of_systems(initial_var) 
        initial_var_str = (' '.join(str(e) for e in initial_var_dict.values())).replace(" ",",")
        
        for _ in range(num_simulations):
          
            text += "{time},{initial_var_str},".format(time=round(time, 2), initial_var_str=initial_var_str)
            update_values = {}

            for key, value in values.items():
                # For printing every value in the data
                text += "{value},".format(value=value) 
            
            
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

                # wether to add comma or not
                if (list(list_keys)[-1] != key):
                    text += "{rate_change},".format(rate_change=rate_change) 
                else:
                    text += "{rate_change}".format(rate_change=rate_change) 

                update_values[key[1]] = values[key[1]]+rate_change*delta

            values = update_values
            
            text += "\n"
            time += delta

        fieldnames = ['t'] + list(initial_var_dict.keys()) + list(values.keys()) + list(self.rate_of_change.keys())
        #print(text)
        if filename != None:
            self.save(text, fieldnames, filename)
        
