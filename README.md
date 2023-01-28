# Thermoelastic Constitutive Artificial Neural Network (thermalCANN)

<p>
<a href="" alt="License:MIT">
    <img src="https://img.shields.io/github/license/llamm-de/thermalCANN" />
</a>
<a href="" alt="License:MIT">
    <img src="https://img.shields.io/badge/Python-3.8.10-blue" />
</a>
<a href="https://github.com/llamm-de/thermalCANN/commits/main" alt="License:MIT">
    <img src="https://img.shields.io/github/last-commit/llamm-de/thermalCANN" />
</a>
</p>

This is the repository for a model of thermo-hyperelasticity using a Constitutive Neural Network (CANN) architecture build with [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/).

The architecture is based on the publications of  [K. Linka et al. (2021): *Constitutive artificial neural networks: A fast and general approach to predictive data-driven constitutive modeling by deep learning*](https://doi.org/10.1016/j.jcp.2020.110010) and [K. Linka & E. Kuhl (2023): *A new family of Constitutive Artificial Neural Networks towards automated model discovery*](https://doi.org/10.1016/j.cma.2022.115731).

## Getting started
This model was developed under Python 3.8.10. The required packages for executing the example scripts can be found in the [requirements file](requirements.txt). To install them into a virtual environment please proceed as follows:
```bash
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the code
### Pretrained model
TBC

### Training the model
TBC

### Running unittests
To run the unit tests provided by this implementation, please use the following 
```bash
python -m unittest
```
from the root directory of this repository. Please be aware that there is not a 100% code coverage. Coverage reports can be generated using the coverage module as 
```
coverage run -m unittest
```

## Datasets
TBC

## Get involved
For feature requests, bug reports etc. please open an issue on github or get in contact with us directly.

## License
This model is licensed under the MIT License. More information on that is given in the [LICENSE.md](LICENSE.md) file.