"""
    Arthur: Adanna Obibuaku
    Purpose: This module is used to reperesent the models defined in the dissertation. In addition,
             simulations of those models
    Date:   09/02/21
"""
from models import State
from models import Automata
from models import AutomataSys
from itertools import product
import numpy as np
from scipy.integrate import odeint
import pandas as pd


def thermostat():
    """
        thermostat:
            This is used for reperesent the thermostat model in our data
    """
    HEATING = State("Heating", lambda temp: temp <= 19 and temp < 23, lambda temp: 26 - temp )
    NO_HEATING = State("No heating", lambda temp: temp >= 23, lambda temp: -0.1*temp )
    thermostat = Automata(NO_HEATING, [HEATING, NO_HEATING], [lambda temp: temp < 19, lambda temp: temp >= 23]) 
    thermostat.run(15, 1, 10000, "data/thermostat.csv")

def newtons_cooling_law():
    """
        newtons_cooling_law:
            This is used for reperesenting the newtons cooling law in our data
    """
    DELTA = 1
    SIMULATIONS = 200

    initial_temp = [i for i in range(1, 61)]
    for t_0 in initial_temp:
        COOLING = State("Cooling", lambda temp: True, lambda temp: -0.015*(temp - 22))
        newtons = Automata(COOLING, [COOLING], [lambda temp: True], t_0)
        newtons.run(t_0, DELTA, SIMULATIONS, "data/train/newtons_cooling_law.csv")


def simple_model_x0():
    """
        This is a very simple model to test the neural network and 
        see wether it performs correctly
    """
    MODEL = State("None", lambda n: True, lambda x: 0)
    simple_automata = Automata(MODEL, [MODEL], [lambda n: True])

    simple_automata.run(1, 1, 1000, "../data/train/simple_model_x0.csv")
    simple_automata.run(1, 1, 100, "../data/test/simple_model_x0.csv", 1000)

def simple_model_x1y2():
    """
        simple_model:
            This is a simple models to test the neural network
    """
    MODEL_1 = State("None", lambda n: True, lambda x: 1)
    simple_automata_1 = Automata(MODEL_1, [MODEL_1], [lambda n: True])

    MODEL_2 = State("None", lambda n: True, lambda y: 2)
    simple_automata_2 = Automata(MODEL_2, [MODEL_2], [lambda n: True])


    simple_automata_1.run(0, 1, 2000, "../data/train/simple_model_x1.csv")
    simple_automata_2.run(0, 1, 2000, "../data/train/simple_model_y2.csv")

def van_der_pol_oscillator(delta, save):
    """
        van der pol oscillator:
            This is used for running a simulation of the van der pol oscillator model
    """
    MU = 0.5
    def f(state, t):
        x, y = state
        dxdt = y
        dydt = MU*(1 - x * x) * y - x
        return dxdt, dydt

    van_df = []
    #100.1  
    time = np.arange(0, 20.1, delta)

    for init_x in range(1,5):
        for init_y in range(1,5):
            states_0 = [init_x, init_y]
            state = odeint(f, states_0, time)

            df = pd.DataFrame(data={'time' : time, 'x' : state[:, 0], 'y' : state[:, 1]})
            df['initial_x'] = init_x
            df['initial_y'] = init_y
            van_df.append(df)

    van_df = pd.concat(van_df)
    if save:
        van_df.to_csv("data/train/van.csv", index = False)
    return van_df



def laub_loomis(delta, save):
    """
        laub_loomis:
            This is used for reperesenting the laun loomis be
    """

    def f(state, t):
        x,y,z,w,p,q,m = state
        func_1 = 1.4 * z - 0.9 * x
        func_2 = 2.5 * p - 1.5 * y
        func_3 = 0.6 * m - 0.8 * y * z
        func_4 = 2 - 1.3 * z * w
        func_5 = 0.7 * x - w * p
        func_6 = 0.3 * x - 3.1 * q
        func_7 = 1.8 * q - 1.5 * y * m
        return func_1, func_2, func_3, func_4, func_5, func_6, func_7

    MIN = 1
    MAX = 3
    STEP = 1

    ranges = range(MIN,MAX,STEP)

    laub_loomis = []
    time = np.arange(0, 500, delta)

    for x, y, z, w, p, q, m in product(ranges, ranges, ranges, ranges, ranges, ranges, ranges):
        states_0 = [x, y, z, w, p, q, m]
        state = odeint(f, states_0, time)
        data = {'time' : time, 'x' : state[:, 0], 'y' : state[:, 1], 'z' : state[:, 2],
                'w' : state[:, 3], 'p' : state[:, 4], 'q' : state[:, 5], 'm' : state[:, 6]}
        df = pd.DataFrame(data=data)
        df['initial_x'] = x
        df['initial_y'] = y
        df['initial_z'] = z
        df['initial_w'] = w
        df['initial_p'] = q
        df['initial_q'] = p
        df['initial_m'] = m
        laub_loomis.append(df)
    
    laub_loomis = pd.concat(laub_loomis)
    if save:
        laub_loomis.to_csv("data/train/laub.csv", index = False)
    return laub_loomis

def lorenz_system(delta, save):
    """
        This runs the lorenz system model using euler method
    """
    SIGMA = 10
    BETA = 8/3
    RHO = 28
    
    def f(state, t):
        x, y, z = state
        dxdt = SIGMA * (y - x)
        dydt = x * (RHO - z) - y
        dzdt = x * y - BETA * z
        return dxdt, dydt, dzdt

    time = np.arange(0, 50.1, delta)
    lorenz = []
    filename = 'data/train/lorenz.csv'

    ranges = range(1,5,1)
    for init_x, init_y, init_z in product(ranges, ranges, ranges): 
        states_0 = [init_x, init_y, init_z]
        state = odeint(f, states_0, time)
        data = {'time' : time, 'x' : state[:, 0], 'y' : state[:, 1], 'z' : state[:, 2],}
        df = pd.DataFrame(data=data)

        df['initial_x'] = init_x
        df['initial_y'] = init_y
        df['initial_z'] = init_z

        lorenz.append(df)

    lorenz =  pd.concat(lorenz)
    if save:
        lorenz.to_csv(filename, index=False)
    return lorenz

if __name__ == "__main__":
    van_der_pol_oscillator(0.001, True)
    laub_loomis(0.1, True)
    lorenz_system(0.01, True)