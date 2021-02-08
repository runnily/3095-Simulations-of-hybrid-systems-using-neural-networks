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

    def __init__(self, current, states, guards):
        """
        __init__: This is used to inilise the hybrid automaton object

        Args:
            current (<class 'state.State'>): This reperesent the current state of the hybrid automaton
            states (<class 'list'>): This reperesents a list of states used within the hybrid automaton
            guards (<class 'list'>): This reperesent a list of guards used with the hybrid automaton

        """
        self.states = states
        self.guards = guards
        self.current = current

        self.lab = {}
        for i in range(len(guards)):
            self.lab[guards[i]] = states[i]
     
    
    def transitions(self, x):
        """
        transitions: This acts as a discerete event, where a guard condition is met we jump to the assiocated 
                     state the guard is pointing to. When the discrete change modeled by x is possible and 
                     what the possible updates of the variables are when the hybrid system makes the discrete change.
        Args:
            x (float): Reperesents the dynamic input
        """
        for guard in self.guards: 
            if guard(x) and self.lab[guard].invariant(x):
                    self.current = self.lab[guard]


    def run(self, y0, delta, num_simulations, filename):
        """
        run: The evolution of the state of the hybrid system over time. This run is achieved
             by performing the euler method

        Args:
            y0 (float): The inital state of the system at the beginning.
            delta (float): The increments we should change by
            noOfSimulations (int): How many numertical simulations we want
        """
        y = y0
        x = 0
        text = ""
        for _ in range(num_simulations):
            dydx = self.current.behaviour(y) # Get the change rate of change
            self.transitions(y) # To change the state if needed
            #print(x,y,dydx) 
            text += "{x},{y},{dydx},{state}\n".format(x=x, y=y,dydx=dydx, state=self.current.name)
            y += dydx*delta # update the change
            x += delta # delta

        with open(filename, "r+") as file:
            writer = csv.DictWriter(file, fieldnames=["x", "y", "dy/dx", "state"])
            try: 
                csv.Sniffer().has_header(file.read(2048))
            except:
                writer.writeheader()
            file.write(text)
            
            

    
