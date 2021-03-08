"""
    Arthur: Adanna Obibuaku
    Purpose: This module is used to reperesent the models
"""
from state import State
from automata import Automata
from van_automata import VanAutomata

def thermostat():
    """
        Thermostat:
            This is used for reperesent the thermostat model in our data
    """
    HEATING = State("Heating", lambda temp: temp <= 19 and temp < 23, lambda temp: 26 - temp )
    NO_HEATING = State("No heating", lambda temp: temp >= 23, lambda temp: -0.1*temp )
    thermostat = Automata(NO_HEATING, [HEATING, NO_HEATING], [lambda temp: temp < 19, lambda temp: temp >= 23]) 
    thermostat.run(15, 1, 10000, "Data/thermostat.csv")

def newtons_cooling_law():
    """
        newtons_cooling_law:
            This is used for reperesenting the newtons cooling law in our data
    """

    initial_temp = [i for i in range(1, 61)]
    env_temp = [i for i in range(1, 61)]
    DELTA = 1
    SIMULATIONS = 200
    for t_0 in initial_temp:
        for t_env in env_temp:
            COOLING = State("Cooling", lambda temp: True, lambda temp: -0.015*(temp - t_env ))
            newtons = Automata(COOLING, [COOLING], [lambda temp: True], t_0, t_env)
            newtons.run(t_0, DELTA, SIMULATIONS, "data/cooling.csv")

def van_der_pol_oscillator():
    """
        newtons_cooling_law:
            This is used for reperesenting the newtons cooling law in our data
    """
    OSICILLATE = State("oscillate", lambda temp: True, lambda x: x*1 )
    van = VanAutomata(OSICILLATE, [OSICILLATE], [lambda temp: True])
    van.run(1, 1, 10, 500)

def very_simple_model():
    """
        This is a very simple model to test the neural network and 
        see wether it performs correctly
    """

    MODEL = State("None", lambda n: True, lambda x: 0)
    simple_automata = Automata(MODEL, [MODEL], [lambda n: True])
    simple_automata.run(1, 1, 200, "data/very_simple_model.csv")


def simple_model():
    """
        simple_model:
            This is a simple models to test the neural network
    """
    MODEL_1 = State("None", lambda n: True, lambda x: 1)
    MODEL_2 = State("None", lambda n: True, lambda y: 2)

    simple_automata = Automata(MODEL_1, [MODEL_1], [lambda n: True])
    simple_automata.run(0, 1, 200, "data/simple_model_1.csv")

    simple_automata = Automata(MODEL_2, [MODEL_2], [lambda n: True])
    simple_automata.run(0, 1, 200, "data/simple_model_2.csv")


if __name__ == "__main__":
    simple_model()
    very_simple_model()
    #newtons_cooling_law()
    #van_der_pol_oscillator()

