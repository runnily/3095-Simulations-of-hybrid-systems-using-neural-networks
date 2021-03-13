"""
    Arthur: Adanna Obibuaku
    Purpose: This module is used to reperesent the models
"""
from state import State
from automata import Automata
from automata_sys import AutomataSys
import pandas as pd


def thermostat():
    """
        Thermostat:
            This is used for reperesent the thermostat model in our data
    """
    HEATING = State("Heating", lambda temp: temp <= 19 and temp < 23, lambda temp: 26 - temp )
    NO_HEATING = State("No heating", lambda temp: temp >= 23, lambda temp: -0.1*temp )
    thermostat = Automata(NO_HEATING, [HEATING, NO_HEATING], [lambda temp: temp < 19, lambda temp: temp >= 23]) 
    thermostat.run(15, 1, 10000, "Data/thermostat.csv")

def newtons_cooling_law(test):
    """
        newtons_cooling_law:
            This is used for reperesenting the newtons cooling law in our data
    """
    DELTA = 1
    SIMULATIONS = 200
    if test:
        initial_temp = [i for i in range(1, 61)]
        for t_0 in initial_temp:
            COOLING = State("Cooling", lambda temp: True, lambda temp: -0.015*(temp - 22))
            newtons = Automata(COOLING, [COOLING], [lambda temp: True], t_0)
            newtons.run(t_0, DELTA, SIMULATIONS, "../data/train/newtons_cooling_law.csv")
    else:
        # This will produce test data from tempreture ranging 61-100, starting t(0) = initial = temp
        initial_temp = [i for i in range(61, 101)]
        for t_0 in initial_temp:
            COOLING = State("Cooling", lambda temp: True, lambda temp: -0.015*(temp - 22))
            newtons = Automata(COOLING, [COOLING], [lambda temp: True], t_0)
            newtons.run(t_0, DELTA, SIMULATIONS, "../data/test/newtons_cooling_law.csv")
    
        # This will produce carrying on from the simulations from the training set. 
        # It will carry on from where each simulation for a inital temp stopped at (199).
        # And carry on getting the rest of the simulation
        initial_temp = [i for i in range(1, 10)]
        for t_0 in initial_temp:
            COOLING = State("Cooling", lambda temp: True, lambda temp: -0.015*(temp - 22))
            newtons = Automata(COOLING, [COOLING], [lambda temp: True], t_0)
            df = pd.read_csv("../data/train/newtons_cooling_law.csv", usecols=[0,1,2])
            df = pd.DataFrame(data=df)
            #df[df.x_0 == t_0][df.x == 199].y.item()

            # Gets value from training set. This is to carry on from the simulations for the training set.
            data = df.query("x_0 == {} & x == {}".format(t_0, 199)).y.item() 
            newtons.run(data, DELTA, SIMULATIONS+1, "../data/test/newtons_cooling_law.csv", 199)


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
        newtons_cooling_law:
            This is used for reperesenting the newtons cooling law in our data
    """
    OSICILLATE = State("oscillate", lambda temp: True, [lambda y: y, lambda x, y: 0.5*(1 - x*x)*y-x])
    van = AutomataSys(OSICILLATE, [OSICILLATE], [lambda temp: True])

    for init_x in range(1,5):
        for init_y in range(1,5):
            van.run([init_x,init_y], 0.1, 500, '../data/train/van.csv')

def coupled_van_der_pol():
    """
        coupled_van_der_Pol:
            This is used for reperesenting the newtons cooling law in our data
    """
    OSICILLATE = State("oscillate", lambda temp: True, [lambda y: y, lambda x, y, z: 0.5*(1 - x*x)*y-2*x+z, lambda w : w, lambda w,z,x : 0.5*(1-z*z)*w - 2*z + x])
    van = AutomataSys(OSICILLATE, [OSICILLATE], [lambda temp: True])

    for init_x in range(1,5):
        for init_y in range(1,5):
            for init_z in range(1,5):
                for init_w in range(1,5):
                    van.run([init_x,init_y,init_z,init_w], 0.1, 500, '../data/train/coupled_van.csv')


def laub_loomis():
    """
        Laub-Loomis benchmark:
            This is used for reperesenting the laun loomis be

    """
    func_1 = lambda z, x : 1.4 * z - 0.9 * x
    func_2 = lambda p, y : 2.5 * p - 1.5 * y
    func_3 = lambda m, y, z : 0.6 * m - 0.8 * y * z
    func_4 = lambda z, w : 2 - 1.3 * z * w
    func_5 = lambda x, w, p : 0.7 * x - w * p
    func_6 = lambda x, q : 0.3 * x - 3.1 * q
    func_7 =  lambda q, y, m : 1.8 * q - 1.5 * y * m
    functions = [func_1, func_2, func_3, func_4, func_5, func_6, func_7]
    LOOMIS = State("loomis", lambda temp: True, functions)
    van = AutomataSys(LOOMIS, [LOOMIS], [lambda temp: True])

    for init_x in range(1,5):
        for init_y in range(1,5):
            for init_z in range(1,5):
                for init_w in range(1,5):
                    for init_p in range(1,5):
                        for init_q in range(1,5):
                            for init_m in range(1,5):
                                van.run([init_x,init_y,init_z,init_w, init_p, init_q, init_m], 0.1, 500, '../data/train/laub_loomis.csv')




if __name__ == "__main__":
    laub_loomis()
