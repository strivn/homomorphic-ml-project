# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup
```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Development Commands
- Run notebooks: `jupyter notebook`
- Train models: Run individual notebook cells
- Save models: `torch.save(model.state_dict(), 'checkpoints/model_name.pth')`

## Code Style Guidelines
- **Classes**: Use CamelCase for class names (e.g., `SimpleClassifier`)
- **Variables/Functions**: Use snake_case (e.g., `train_loader`, `evaluate_model`)
- **Imports**: Order as: standard library → PyTorch → ML libraries → domain-specific
- **Type Hints**: Add when introducing new functions
- **Formatting**: Use 4-space indentation
- **Models**: Follow PyTorch nn.Module patterns with proper init/forward methods
- **Error Handling**: Use assertions for validating tensor dimensions
- **Comments**: Document complex operations and architecture choices
- **Homomorphic Encryption**: Test both plaintext and encrypted versions

This project uses PyTorch with homomorphic encryption libraries (Concrete ML, TenSEAL) for privacy-preserving machine learning. The end goal is to compare and understand nuances of using privacy enhancing technologies, so the notebook should be set up to experiment and understand performance like accuracy, time, etc. Include helpful inline comments. Use web search to get the API references you need.
