"""
    Arthur: Adanna Obibuaku

    Purpose: This module is used to reperesent the models
"""
from state import State
from automata import Automata

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
    COOLING = State("Cooling", lambda temp: True, lambda temp: -0.015*(temp-22))
    newtons = Automata(COOLING, [COOLING], [lambda temp: True])
    newtons.run(50, 1, 500, "Data/cooling.csv")

if __name__ == "__main__":
    #newtons_cooling_law()
    thermostat()

