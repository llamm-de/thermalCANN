import os
import pickle

def load_treloar():
    """
    Load dataset from TRELOAR (1944) as numpy arrays
    """
    path = os.path.join(os.getcwd(), 'data/Treloar/Treloar_steinmann.pkl')
    with open(path, 'rb') as f:
        uniaxial_data, pure_shear_data, equibiaxial_data = pickle.load(f) 

    return uniaxial_data, pure_shear_data, equibiaxial_data
