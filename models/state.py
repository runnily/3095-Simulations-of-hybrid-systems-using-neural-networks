"""
Arthur:
    Adanna Obibuaku

Purpose:
    This module is used for building a state of a hybrid class.
    This module is commonly going to be called within the hybrid
    automata module
"""

class State:
    """
    State: This class is used for reperesenting a state in a hybrid automaton
    """
    def __init__(self, name, invariant,  behaviour):
        """
            Args:
                name (string): This is the name of the state
                invariant (<class 'function'>): This defines the invarient of the state
                behavour (<class 'function'>): This defines the behaviour of the state
        """
        self.name = name
        self.invariant = invariant
        self.behaviour = behaviour
    
    