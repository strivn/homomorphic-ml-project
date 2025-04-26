# Homomorphic Encryption for Privacy Preserving ML

## Overview
This project experiments with the use of homomorphic encryption (HE) in machine learning to enable privacy-preserving machine learning (specifically with MLP and CNN). Homomorphic encryption unlocks the ability to perform computations on encrypted data without decryption, and one example of this is enabling organizations use cloud services without providing any plain data. This became our motivation to explore homomorphic encryption in this project. 

We use three key libraries for the experiments:

- PyTorch: For building standard neural network models
- Concrete ML (by Zama): For compiling PyTorch models to FHE-compatible versions
- TenSEAL (by OpenMined): For encrypted tensor operations and neural networks

Homomorphic encryption allows computations to be performed directly on encrypted data without decryption, enabling privacy-preserving inference where sensitive data remains protected throughout the entire process.
This project is completed as part of coursework for 95-878 Engineering Privacy in Software.

## Main Notebooks
- [PoC_Revised](0-poc_revised.ipynb): Initial testing with very simple MLP
- [AdultIncomeMLP](1-PathMNIST.ipynb): Tenseal and Concrete PTQ experiment with MLP on Adult Income dataset
- [MNISTPtq](2-PathMNIST.ipynb): Concrete PTQ experiment with MLP on MNIST 
- [MNISTQat](2-PathMNIST.ipynb): Concrete QAT experiment with MLP on MNIST 
- [PathMNIST](3-PathMNIST.ipynb): Concrete PTQ experiment with CNN on PathMNIST


## Repository Structure

- `archived/`         # Previous versions and experiments
- `checkpoints/`      # Model checkpoints
- `papers/`           # Reference papers and documentation
- `tutorials/`        # Tutorials from OpenMined and Zama
- `utils/`            # Utility functions and helper modules
- `requirements.txt`  # Python dependencies
- `README.md `        # Project documentation

## Quick Start

### Setup Environment
```bash
# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Data
Download PathMNIST from: [Drive](https://drive.google.com/file/d/1nUePNjO8V3z2VehnFjlTljPMeILVhKeR/view?usp=share_link)
Unzip and store to ./data

