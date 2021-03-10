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
            
            if (self.x_0 == None) and (self.x_env == None):
                text += '{x},{y},{dydx},"{state}"\n'.format(x=x, y=y,dydx=dydx, state=self.current.name)
                fieldnames = ["x", "y", "dy/dx", "state"]
            elif (self.x_0 != None) and (self.x_env == None):
                text += '{x_0},{x},{y},{dydx},"{state}"\n'.format(x_0 = self.x_0, x=x, y=y,dydx=dydx, state=self.current.name)
                fieldnames = ["x_0", "x", "y", "dy/dx", "state"]
            elif (self.x_0 == None) and (self.x_env != None):
                text += '{x_env},{x},{y},{dydx},"{state}"\n'.format(x_env = self.x_env, x=x, y=y,dydx=dydx, state=self.current.name)
                fieldnames = ["x_env", "x", "y", "dy/dx", "state"]
            else:
                text += '{x_0},{x_env},{x},{y},{dydx},"{state}"\n'.format(x_0 = self.x_0, x_env = self.x_env, x=x, y=y,dydx=dydx, state=self.current.name)
                fieldnames = ["x_0", "x_env", "x", "y", "dy/dx", "state"]

            y += dydx*delta # update the change
            x += delta # delta

        self.save(text, fieldnames, filename)
            
            

    
