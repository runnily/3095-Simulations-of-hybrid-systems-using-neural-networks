"""
    Arthur: Adanna Obibuaku

    Purpose: This module is used to reperesent the models
"""
from state import State
from automata import Automata

def thermostat():
    HEATING = State("Heating", lambda temp: temp <= 19 and temp < 23, lambda temp: 26 - temp )
    NO_HEATING = State("No heating", lambda temp: temp >= 23, lambda temp: -0.1*temp )
    thermostat = Automata(NO_HEATING, [HEATING, NO_HEATING], [lambda temp: temp < 19, lambda temp: temp >= 23]) 
    thermostat.run(15, 1, 10000, "Data/heating.csv")

if __name__ == "__main__":
    thermostat()

