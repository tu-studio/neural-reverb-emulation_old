import torch
from network.encoder import EncoderTCN
from network.decoder import DecoderTCN
from network.training import train
from network.evaluate import evaluate
from torch.utils.data import  random_split, DataLoader
from network.dataset import AudioDataset
from network.metrics import spectral_distance
import random 
import numpy as np
import torchinfo
from utils import logs, config
import os
from pathlib import Path
import ast

def prepare_device(request):
    if request == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
        else:
            device = torch.device("cpu")
            print("MPS requested but not available. Using CPU device")
    elif request == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device")
        else:
            device = torch.device("cpu")
            print("CUDA requested but not available. Using CPU device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

def set_random_seed(random_seed):
    if 'random' in globals():
        random.seed(random_seed)
    else:
        print("The 'random' package is not imported, skipping random seed.")

    if 'np' in globals():
        np.random.seed(random_seed)
    else:
        print("The 'numpy' package is not imported, skipping numpy seed.")

    if 'torch' in globals():
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(random_seed)
    else:
        print("The 'torch' package is not imported, skipping torch seed.")
    
    if 'tf' in globals():
        tf.random.set_seed(random_seed)
    else:
        print("The 'tensorflow' package is not imported, skipping tensorflow seed.")

    if 'scipy' in globals():
        scipy.random.seed(random_seed)
    else:
        print("The 'scipy' package is not imported, skipping scipy seed.")
    
    if 'sklearn' in globals():
        sklearn.utils.random.seed(random_seed)
    else:
        print("The 'sklearn' package is not imported, skipping sklearn seed.")

def main():
    # Load the hyperparameters from the params yaml file into a Dictionary
    params = config.Params('params.yaml')

    n_inputs = params["train"]["n_inputs"]
    n_bands = params["train"]["n_bands"]
    latent_dim = params["train"]["latent_dim"]
    n_epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    kernel_size = params["train"]["kernel_size"]
    n_blocks = params["train"]["n_blocks"]
    dilation_growth = params["train"]["dilation_growth"]
    n_channels = params["train"]["n_channels"]
    lr = params["train"]["lr"]
    use_kl = params["train"]["use_kl"]
    device_request = params["train"]["device_request"]
    random_seed = params["general"]["random_seed"]
    input_file = params["train"]["input_file"]

    # Define and create the path to the tensorboard logs directory in the source repository
    default_dir = config.get_env_variable('DEFAULT_DIR')
    dvc_exp_name = config.get_env_variable('DVC_EXP_NAME')   
    tensorboard_path = Path(f'{default_dir}/logs/tensorboard/{dvc_exp_name}')
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    # Create a SummaryWriter object to write the tensorboard logs
    writer = logs.CustomSummaryWriter(log_dir=tensorboard_path)

    # Add hyperparameters and metrics to the hparams plugin of tensorboard
    metrics = {}
    writer.add_hparams(hparam_dict=params.flattened_copy(), metric_dict=metrics, run_name=tensorboard_path)

    # Set a random seed for reproducibility across all devices
    set_random_seed(random_seed)

    # Prepare the requested device for training. Use cpu if the requested device is not available 
    device = prepare_device(device_request)

        # Build the model
    encoder = EncoderTCN(
        n_inputs=n_bands,
        kernel_size=kernel_size, 
        n_blocks=n_blocks, 
        dilation_growth=dilation_growth, 
        n_channels=n_channels,
        latent_dim=latent_dim,
        use_kl=use_kl)
    
    decoder = DecoderTCN(
        n_outputs=n_bands,
        kernel_size=kernel_size,
        n_blocks=n_blocks, 
        dilation_growth=dilation_growth, 
        n_channels=n_channels,
        latent_dim=latent_dim,
        use_kl=use_kl)
    
    # setup loss function, optimizer, and scheduler
    criterion = spectral_distance

    # Setup optimizer
    model_params = list(encoder.parameters())
    model_params += list(decoder.parameters())
    optimizer = torch.optim.Adam(model_params, lr, (0.5, 0.9))

    # TODO: Implement this
    # # Add the model graph to the tensorboard logs
    # sample_inputs = torch.randn(1, 1, input_size) 
    # writer.add_graph(model, sample_inputs.to(device))

    # Load the dataset
    full_dataset = AudioDataset(input_file, apply_augmentations=False)

      # Define the sizes of your splits
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Get the sample rate
    sample_rate = full_dataset.get_sample_rate()

    # Create the splits
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    train(encoder, decoder, train_loader, val_loader, criterion, optimizer, tensorboard_writer=writer, num_epochs=n_epochs, device=device, n_bands=n_bands, use_kl=use_kl, sample_rate=sample_rate)

    # Evaluate the model
    evaluate(encoder, decoder, test_loader, criterion, writer, device, n_bands, use_kl, sample_rate)

    if not os.path.exists('exp-logs/'):
        os.makedirs('exp-logs/')

    writer.close()

    # TODO: Implement this
    # # Save the model checkpoint
    # output_file_path = Path('model/checkpoints/network.pth')
    # output_file_path.parent.mkdir(parents=True, exist_ok=True)
    # torch.save(network.state_dict(), output_file_path)
    # print("Saved PyTorch Model State to network.pth")

    print("Done with the training stage!")

if __name__ == "__main__":
    main()
