import os
import pickle

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


def load_artificial_Arruda_Boyce():
    """
    Load dataset of artificially generated thermo-elastic data (incompressible) for uniaxial tension generated from Arruda & Boyce model.

    Set contains:
        1. Stretches
        2. Temperature delta (T - T_ref)
        3. First Piola Kirchhoff stress in loading direction
    """

    with open(os.path.join(os.getcwd(),'data/artificially_generated/uniaxial_Arruda_Boyce.pkl'), 'rb') as f:
        stretch, delta_theta, stress = pickle.load(f)

    return stretch, delta_theta, stress