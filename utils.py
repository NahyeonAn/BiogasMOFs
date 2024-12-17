import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import pyapep.isofit as isof
import pyapep.simide as simi

import matplotlib.pyplot as plt
import numpy as np

parameters = {'axes.labelsize': 17,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
          'axes.titlesize': 20}
plt.rcParams.update(parameters)
plt.rcParams['font.family'] = 'arial'

from sklearn.metrics import r2_score
from scipy.optimize import minimize

# A function that creates Python functions for isotherm models based on their names and parameters
def MakeIsoFunc(func, par):
    def Lang(p, T):
        num = par[0]*par[1]*p
        den = 1+par[1]*p
        return num/den

    def Freu(p, T): # Freundlich isotherm model
        q = par[0]*p**par[1]
        return q

    ### With 3 parameters ###
    def Quad(p, T): # Quadratic isotherm model
        bP = par[1]*p
        dPP = par[2]*p**2
        deno = 1+ bP + dPP
        nume = par[0]*(bP + 2*dPP)
        q = nume/deno
        return q

    def Sips(p, T): # Sips isotherm model 
        n = par[2]
        numo = par[0]*par[1]*p**n
        deno = 1 + par[1]*p**n
        q = numo/deno
        return q
    
    def Red(p, T): #Redlich-Petersin
        num = par[0]*p
        den = 1+par[1]*(p**par[2])
        return num/den

    def Toth(p, T): #Toth
        num = par[0]*p
        den = (par[1]+(p)**(1/par[2]))**(par[2])
        return num/den
    
    def Gab(p, T):
        numo = par[0]*par[1]*par[2]*p
        deno = (1-par[2]*p)*(1+(par[1]-1)*par[2]*p)
        q = numo/deno
        return q
    
    def Pel(p, T): #Peleg
        q = par[0]*(p**par[1])+par[2]*(p**par[3])
        return q
    
    def DA(p, T): #Dubinin–Astakhov equation
        q = par[0]*np.exp(-(par[1]*np.log(par[2]/p))**par[3])
        return q
    
    if func == 'Lang':
        isotherm = Lang
    elif func == 'Freu':
        isotherm = Freu
    elif func == 'Quad':
        isotherm = Quad
    elif func == 'Sips':
        isotherm = Sips
    elif func == 'Red':
        isotherm = Red
    elif func == 'Toth':
        isotherm = Toth
    elif func == 'Gab':
        isotherm = Gab
    elif func == 'Pel':
        isotherm = Pel
    elif func == 'DA':
        isotherm = DA
    
    return isotherm

# A function that derives CO2 and CH4 isotherm functions from the given data using MakeIsoFunc
def WhichIsoFunction(target):
    #CO2
    gases = ['CO2', 'CH4']
    iso_mix = []
    for gas in gases:
        col_name = [f'{gas}_Isotherm_{i}' for i in range(3,0,-1)]
        for col in col_name:
            if target[col] != 'NaN':
                idx = col[-1]
                if col == col_name[-1]:
                    par_ = target.loc[[f'par_1_{gas}', f'par_2_{gas}']].values
                    func_ = target[col]
                    
                else:
                    par_str = target[f'par_{gas}_{idx}'][1:-1].split()
                    par_ = [float(par) for par in par_str]
                    func_ = target[col]
                iso_ = MakeIsoFunc(func_, par_)
                break
            else:
                continue
        iso_mix.append(iso_)
    return iso_mix