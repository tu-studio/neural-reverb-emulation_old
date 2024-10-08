import torch
from utils import config
from train import get_train_mode_params
from model import NeuralNetwork
from pathlib import Path

def main():
    params = config.Params()
    input_size = params['general']['input_size']
    train_mode = params['train']['train_mode']

    # Get model hyperparameters based on training mode
    _, conv1d_strides, conv1d_filters, hidden_units = get_train_mode_params(train_mode)

    # Define the model (ensure to use the same architecture as in training.py)
    model = NeuralNetwork(conv1d_filters, conv1d_strides, hidden_units)

    # Load the model state
    input_file_path = Path('model/checkpoints/network.pth')
    network.load_state_dict(torch.load(input_file_path, map_location=torch.device('cpu')))

    # Export the model
    example = torch.rand(1, 1, input_size)
    output_file_path = Path('model/exports/network.onnx') 
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example, output_file_path, export_params=True, opset_version=17, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print("Model exported to ONNX format.")

if __name__ == "__main__":
    main()
