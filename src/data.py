import os
import pickle
import numpy as np
from scipy.optimize import curve_fit

def load_treloar(arruda_boyce_fit=False):
    """
    Load dataset from TRELOAR (1944) as numpy arrays.
    
    For an extended dataset recovered from a fit of the Arruda&Boyce model, 
    use arruda_boyce_fit=True.
    """

    if arruda_boyce_fit:
        path = os.path.join(os.getcwd(), 'data/Treloar/Treloar_Arruda_Boyce.pkl')
    else:    
        path = os.path.join(os.getcwd(), 'data/Treloar/Treloar_steinmann.pkl')
    
    with open(path, 'rb') as f:
        uniaxial_data, pure_shear_data, equibiaxial_data = pickle.load(f) 

    return uniaxial_data, pure_shear_data, equibiaxial_data
