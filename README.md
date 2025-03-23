# Homomorphic Encryption for Privacy Preserving ML

## Overview
This project demonstrates the application of homomorphic encryption in deep learning using PyTorch, Concrete ML (by Zama), and TenSEAL (by OpenMined). 
It implements a convolutional neural network architecture that works with both plaintext data and homomorphically encrypted data for privacy-preserving machine learning.

This project is completed as part of coursework for **95-878 Engineering Privacy in Software**.

## Main Notebooks
- [PoC_Revised](0-poc_revised.ipynb): initial testing with very simple neural network
- [PathMNIST](1-PathMNIST.ipynb): testing with PathMNIST dataset with simple convolutional network


## Repository Structure
- `.data/`
- `archived/` 
- `tutorials/`: relevant tutorials from OpenMined and Zama
- `utils/`


## Quick Start

### Setup Environment
```bash
# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

Download PathMNIST from: https://drive.google.com/file/d/1nUePNjO8V3z2VehnFjlTljPMeILVhKeR/view?usp=share_link
Unzip and store to ./data