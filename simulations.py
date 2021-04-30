"""
    Arthur: Adanna Obibuaku
    Purpose: This module is used to reperesent the models defined in the dissertation. In addition,
             simulations of those models
    Date:   09/02/21
"""
from itertools import product
import numpy as np
from scipy.integrate import odeint
import pandas as pd

def newtons_cooling_law(delta, simulations, save, initial_paras = None):
    """
        newtons_cooling_law:
            This is used for reperesenting the newtons cooling law in our data
    """
   
    def f(state, t):
        temp = state
        dtempdtime = -0.015*(temp - 22)
        return dtempdtime
    initial_temp = [i for i in range(1, 61)]

    initial_temp = range(1, 61, 1) # we want initial parameters to be x1, x2, x2 ... x9 (0.99, 1.01)

    try:
        if (initial_paras != None).any():
            initial_temp = initial_paras
    except AttributeError:
        pass

    newton = []
    time = np.arange(0, simulations, delta)

    for t_0 in initial_temp:
        states_0 = [t_0]
        state = odeint(f, states_0, time)

        df = pd.DataFrame(data={'time' : time, 'temp' : state[:, 0]})
        df['initial_temp'] = t_0
        newton.append(df)
    
    newton = pd.concat(newton)
    if save:
        newton.to_csv("data/train/newtons_cooling_law.csv", index = False)   
    return newton

def van_der_pol_oscillator(delta, simulations, save, initial_paras = None):
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
    time = np.arange(0, simulations, delta)

    ranges = range(1,5)

    try:
        if (initial_paras != None).any():
            ranges = initial_paras
    except AttributeError:
        pass


    for init_x in ranges:
        for init_y in ranges:
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



def laub_loomis(delta, simulations, save, initial_paras = None):
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
    try:
        if (initial_paras != None).any():
            ranges = initial_paras
    except AttributeError:
        pass


    laub_loomis = []
    time = np.arange(0, simulations, delta) # 0 , 500

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

def biological_model(delta, simulations, save, initial_paras = None):
    """
        This repersents a biological model
    """
    def f(state, t):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = state
        dx1dt = 3 * x3 - x1 * x6 
        dx2dt = x4 - x2 * x6
        dx3dt = x1 * x6 - 3 * x3
        dx4dt = x2 * x6 - x4
        dx5dt = 3 * x3 + 5 * x1 - x5 
        dx6dt = 5 * x5 + 3 * x3 + x4 - x6 * (x1 + x2 + 2 * x8 + 1)
        dx7dt = 5 * x4 + x2 - 0.5 * x7
        dx8dt = 5 * x7 - 2 * x6 * x8 + x9 - 0.2 * x8
        dx9dt = 2 * x6 * x8 - x9
        return dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt, dx8dt, dx9dt
    
    time = np.arange(0, simulations, delta)
    biological_model = []
    filename = 'data/train/biological_model.csv'

    ranges = np.arange(0.99, 1.01, 0.02) # we want initial parameters to be x1, x2, x2 ... x9 (0.99, 1.01)
    try:
        if (initial_paras != None).any():
            ranges = initial_paras
    except AttributeError:
        pass

    for init_x1, init_x2, init_x3, init_x4, init_x5, init_x6, init_x7, init_x8, init_x9 in product(ranges, ranges, ranges, ranges, ranges, ranges, ranges, ranges, ranges):
        states_0 = [init_x1, init_x2, init_x3, init_x4, init_x5, init_x6, init_x7, init_x8, init_x9]
        state = odeint(f, states_0, time)
        data = {'time' : time, 'x1' : state[:, 0], 'x2' : state[:, 1], 'x3' : state[:, 2], 'x4' : state[:, 3],
            'x5' : state[:, 4], 'x6' : state[:, 5], 'x7' : state[:, 6], 'x8' : state[:, 7], 'x9' : state[:, 8],}
        
        df = pd.DataFrame(data=data)
        df['initial_x1'] = init_x1
        df['initial_x2'] = init_x2
        df['initial_x3'] = init_x3
        df['initial_x4'] = init_x4
        df['initial_x5'] = init_x5
        df['initial_x6'] = init_x6
        df['initial_x7'] = init_x7
        df['initial_x8'] = init_x8
        df['initial_x9'] = init_x9
        biological_model.append(df)

    biological_model =  pd.concat(biological_model)
    
    if save:
        biological_model.to_csv(filename, index=False)
    return biological_model

if __name__ == "__main__":
    newtons_cooling_law(delta = 1, simulations = 200 , save = False, initial_paras = np.random.uniform(0, 60, 300))