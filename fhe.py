import time
import torch
import numpy as np
from concrete.fhe import Configuration
from concrete.ml.torch.compile import compile_torch_model
from torchvision import transforms
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from concrete.ml.torch.compile import compile_brevitas_qat_model


batch_size = 64
learning_rate = 0.001
hidden_size = 32
num_epochs = 10
input_size = 28 * 28  # MNIST images are 28x28 pixels
num_classes = 10  # 10 digits (0-9)

device = "cpu"


import brevitas.nn as qnn
import torch.nn as nn
import torch

class QuantMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, n_bits=3):
        super(QuantMLP, self).__init__()

        self.flatten = nn.Flatten()

        # Input quantization
        self.quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)

        # First quantized linear layer
        self.fc1 = qnn.QuantLinear(input_size, hidden_size, True,
                                   weight_bit_width=n_bits, bias_quant=None)

        # Activation quantization after first layer
        self.quant_act1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)

        # Second quantized linear layer
        self.fc2 = qnn.QuantLinear(hidden_size, hidden_size, True,
                                   weight_bit_width=n_bits, bias_quant=None)

        # Activation quantization after second layer
        self.quant_act2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)

        # Output quantized linear layer
        self.fc3 = qnn.QuantLinear(hidden_size, num_classes, True,
                                   weight_bit_width=n_bits, bias_quant=None)

    def forward(self, x):
        x = self.flatten(x)
        x = self.quant_inp(x)
        x = self.quant_act1(torch.relu(self.fc1(x)))
        x = self.quant_act2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    


def scale_layers_differently(model, fc1_scale=0.005, fc2_scale=0.01, fc3_scale=0.02):
    state_dict = model.state_dict()
    
    # Apply different scaling factors to different layers
    for key, value in state_dict.items():
        if 'fc1.weight' in key:
            state_dict[key] = value * fc1_scale
        elif 'fc2.weight' in key:
            state_dict[key] = value * fc2_scale
        elif 'fc3.weight' in key:
            state_dict[key] = value * fc3_scale
    
    # Load the modified state dictionary back into the model
    model.load_state_dict(state_dict)
    
    return model




def main():
    # Load your existing test dataset
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        transform=transform,
        download = True
    )
    
    print("Loading Data")
    # Select 5 random images
    indices = torch.randperm(len(test_dataset))[:1]
    selected_images = torch.stack([test_dataset[i][0] for i in indices])
    selected_labels = torch.tensor([test_dataset[i][1] for i in indices])
    
    # Convert grayscale to RGB
    images = torch.stack([img for img in selected_images])

    print("Loading Model")
    
    # Load your pre-trained model
    model = QuantMLP(input_size, hidden_size, num_classes).to(device)   # MNIST has 10 classes
    model.load_state_dict(torch.load('mnist_mlp.pth'))
    
    torch_input = torch.randn(100, 784)
    # Compile the model for FHE
    print("Compiling model for FHE...")
    q_module = compile_brevitas_qat_model(
    model,  # our model
    torch_input,  # a representative input-set for quantization and compilation
    rounding_threshold_bits={"n_bits": 6, "method": "approximate"}
)
    
    # Run FHE inference
    print("\nRunning FHE inference on 5 images...")
    total_time = 0
    
    for i in range(1):
        print(f"\nImage {i+1}:")
        img = images[i].unsqueeze(0)  # Add batch dimension
        
        # Run FHE execution
        start_time = time.time()
        encrypted_output = q_module.forward(img.numpy(), fhe="execute")
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Get prediction
        predicted_class = np.argmax(encrypted_output)
        print(f"  True label: {selected_labels[i].item()}")
        print(f"  Predicted class: {predicted_class}")
        print(f"  Inference time: {inference_time:.2f} seconds")
    
    print(f"\nTotal inference time for 5 images: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()