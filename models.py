"""
    Arthur: Adanna Obibuaku
    Purpose: This module is used to reperesent the models
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
    thermostat.run(15, 1, 10000, "Data/thermostat.csv")

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
        newtons.run(t_0, DELTA, SIMULATIONS, "../data/train/newtons_cooling_law.csv")


def simple_model_x0(test):
    """
        This is a very simple model to test the neural network and 
        see wether it performs correctly
    """
    MODEL = State("None", lambda n: True, lambda x: 0)
    simple_automata = Automata(MODEL, [MODEL], [lambda n: True])
    if test:
        simple_automata.run(1, 1, 1000, "../data/train/simple_model_x0.csv")
    else:
        simple_automata.run(1, 1, 100, "../data/test/simple_model_x0.csv", 1000)
   


def simple_model_x1y2(test):
    """
        simple_model:
            This is a simple models to test the neural network
    """
    MODEL_1 = State("None", lambda n: True, lambda x: 1)
    simple_automata_1 = Automata(MODEL_1, [MODEL_1], [lambda n: True])

    MODEL_2 = State("None", lambda n: True, lambda y: 2)
    simple_automata_2 = Automata(MODEL_2, [MODEL_2], [lambda n: True])

    if test:
        simple_automata_1.run(0, 1, 2000, "../data/train/simple_model_x1.csv")

        simple_automata_2.run(0, 1, 2000, "../data/train/simple_model_y2.csv")
    else:
        simple_automata_1.run(2000, 1, 100, "../data/test/simple_model_x1.csv", 2000)

        simple_automata_2.run(4000, 1, 100, "../data/test/simple_model_y2.csv",2000)

def van_der_pol_oscillator():
    """
        van der pol oscillator:
            This is used for reperesenting the newtons cooling law in our data
    """
    MU = 0.5
    def f(state, t):
        x, y = state
        dxdt = y
        dydt = MU*(1 - x * x) * y - x
        return dxdt, dydt

    van_df = []
    time = np.arange(0, 200, 0.1)

    for init_x in range(1,5):
        for init_y in range(1,5):
            states_0 = [init_x, init_y]
            state = odeint(f, states_0, time)

            df = pd.DataFrame(data={'time' : time, 'x' : state[:, 0], 'y' : state[:, 1]})
            df['initial_x'] = init_x
            df['initial_y'] = init_y
            van_df.append(df)

    van_df = pd.concat(van_df)
    van_df.to_csv("data/train/van.csv", index = False)



def laub_loomis():
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
    time = np.arange(0, 500, 0.1)

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
    laub_loomis.to_csv("data/train/laub_loomis.csv", index = False)

def lorenz_system():
    """
        This runs the lorenx system model using euler method
    """
    sigma = 10
    beta = 8/3
    rho = 28

    dxdt = lambda y, x : sigma * (y - x)
    dydt = lambda x, y, z: x * (rho - z) - y
    dzdt = lambda x, y, z: x * y - beta * z

    funcs = [dxdt, dydt, dzdt]
    LORENZ = State("lorenz", lambda x: True, funcs)
    system = AutomataSys(LORENZ, [LORENZ], [lambda x: True])


    for init_x in range (1,3):
        for init_y in range(1,3):
            for init_z in range(1,3):
                system.run([init_x, init_y, init_z], 0.01, 3000, 'data/train/lorenz.csv')
    """
    kjnfor init_x in range(5,7):
        for init_y in range(5,7):
            for init_z in range(5,7):
                system.run([init_x, init_y, init_z], 0.01, 25, '/data/train/lorenz.csv')
    
    for init_x in range (1,5):
        for init_y in range(1,5):
            for init_z in range(1,5):
                df = pd.DataFrame(data=pd.read_csv("../data/train/lorenz.csv", usecols=[0,4,5,6]))
                data = df.query("init_x == {} & init_y = {} & init_z = {} & t == {}".format(init_x, init_y, init_z, 999)).y.item() 
                system.run(data, 0.1, 20, "../data/test/lorenz.csv", 999)"""


if __name__ == "__main__":
    van_der_pol_oscillator()