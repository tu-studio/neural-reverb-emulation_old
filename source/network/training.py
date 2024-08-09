import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from network.ravepqmf import PQMF,  center_pad_next_pow_2
from utils import config
import os
from network.metrics import plot_distance_spectrums_tensorboard, plot_spectrums_tensorboard

def train(encoder, decoder, train_loader, val_loader, criterion, optimizer, tensorboard_writer, num_epochs=25, device='cpu', n_bands=64, use_kl=False, sample_rate=44100):
    encoder.to(device)
    decoder.to(device)

    # Initialize PQMF
    pqmf = PQMF(100, n_bands).to(device)

    for epoch in range(num_epochs):
        #Training Loop
        encoder.train()
        decoder.train()


        #initialize epoch losses
        train_epoch_loss = 0
        train_epoch_kl_div = 0
        train_epoch_criterion = 0

        print(f"Train loader shape: {len(train_loader)}")

        for epoch, (dry_audio, wet_audio) in enumerate(train_loader):
            
            print(f"Epoch {epoch}")

            print(f"Dry audio shape: {dry_audio.shape}")
            print(f"Wet audio shape: {wet_audio.shape}")

            # Zero the parameter gradients
            optimizer.zero_grad()

             # Pad both dry and wet audio to next power of 2
            dry_audio = center_pad_next_pow_2(dry_audio)
            wet_audio = center_pad_next_pow_2(wet_audio)

            print(f"Dry audio shape after padding: {dry_audio.shape}")
            print(f"Wet audio shape after padding: {wet_audio.shape}")

            # Apply PQMF to input
            dry_audio_decomposed = pqmf(dry_audio)
            wet_audio_decomposed = pqmf(wet_audio)

            print(f"Dry audio decomposed shape: {dry_audio_decomposed.shape}")
            print(f"Wet audio decomposed shape: {wet_audio_decomposed.shape}")

            for batch in range(dry_audio.shape[0]):

                print(f"Batch {batch}")

                # TODO: This should not be necessary if preprocessing is done correctly
                dry_audio_batch = dry_audio_decomposed[batch]
                wet_audio_batch = wet_audio_decomposed[batch]

                dry_audio_batch = dry_audio_batch.view(1, 1, -1)
                wet_audio_batch = wet_audio_batch.view(1, 1, -1)

                # Throw error if wet audio is longer than dry audio
                if wet_audio_batch.shape[-1] != dry_audio_batch.shape[-1]:
                    raise ValueError(f"Wet audio is not the same length than dry audio: {wet_audio_batch.shape[-1]} vs {dry_audio_batch.shape[-1]}")
            
                dry_audio_batch, wet_audio_batch = dry_audio_batch.to(device), wet_audio_batch.to(device)

                print(f"Dry audio batch shape: {dry_audio_batch.shape}")
                print(f"Wet audio batch shape: {wet_audio_batch.shape}")

                # # Apply PQMF to input
                # dry_audio_decomposed = pqmf(dry_audio_batch)
                # wet_audio_decomposed = pqmf(wet_audio_batch)

                # print(f"Dry audio decomposed shape: {dry_audio_decomposed.shape}")
                # print(f"Wet audio decomposed shape: {wet_audio_decomposed.shape}")

                dry_audio_decomposed = dry_audio_batch
                wet_audio_decomposed = wet_audio_batch
    
                # Forward pass through encoder
                encoder_outputs = []
                x = dry_audio_decomposed

                for block in encoder.blocks:
                    x = block(x)
                    encoder_outputs.append(x)
        
                # Get the final encoder output
                z = encoder_outputs.pop()

                # Reverse the list of encoder outputs for the decoder
                encoder_outputs = encoder_outputs[::-1]
                encoder_outputs.append(dry_audio_decomposed)

                # TODO: This should be done better and we don't care right now
                # # Forward pass through encoder
                # if use_kl:
                #     mu, logvar = encoder(dry_audio_decomposed)
                #     z = encoder.reparameterize(mu, logvar)
                #     kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/ mu.shape[-1]
                #     train_epoch_kl_div += kl_div

                # Forward pass through decoder
                net_outputs_decomposed = decoder(z, encoder_outputs)

                output_decomposed = net_outputs_decomposed + dry_audio_decomposed

                # loss = criterion(output_decomposed, wet_audio_decomposed)

                output = pqmf.inverse(output_decomposed)
                wet = pqmf.inverse(wet_audio_decomposed)

                loss = criterion(output, wet)

                if batch == 0: 
                #     # Assuming x and y are your input tensors
                    plot_spectrums_tensorboard(tensorboard_writer, output, wet, step=0)
                    plot_distance_spectrums_tensorboard(tensorboard_writer, output, wet, step=0)

                # output = output_decomposed

                output = pqmf.inverse(output_decomposed)

                # Check that net outputs are the same length as the dry audio
                if output.shape[-1] != dry_audio_batch.shape[-1]:
                    raise ValueError(f"Net outputs are not the same length as the dry audio {output.shape[-1]} vs {dry_audio_batch.shape[-1]}")
                
                # if use_kl:
                #     loss += kl_div

                # Add KL divergence to the loss
                train_epoch_loss += loss 
                
                train_epoch_criterion += loss

            # Backward pass and optimization
            loss.backward()

            # # Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

            optimizer.step()

        
            train_avg_epoch_loss = train_epoch_loss / len(train_loader)
            train_avg_epoch_loss_criterion = train_epoch_criterion / len(train_loader)  
            if use_kl:
                train_avg_epoch_kl_div = train_epoch_kl_div / len(train_loader)

            # Log loss
            tensorboard_writer.add_scalar("Loss/ training loss", train_avg_epoch_loss, epoch)
            tensorboard_writer.add_scalar("Loss/ training criterion", train_avg_epoch_loss_criterion, epoch)
            if use_kl:
                tensorboard_writer.add_scalar("Loss/training kl_div", train_avg_epoch_kl_div, epoch)
            # Log audio samples
            print(dry_audio.shape)
            print(wet_audio.shape)
            print(output.shape)
            tensorboard_writer.add_audio("Audio/TCN_Input", dry_audio[-1, 0, :].cpu().cpu().squeeze(0), epoch, sample_rate=sample_rate)
            tensorboard_writer.add_audio("Audio/TCN_Target", wet_audio[-1, 0, :].cpu().cpu().squeeze(0), epoch, sample_rate=sample_rate)
            tensorboard_writer.add_audio("Audio/TCN_output", output.cpu().squeeze(0), epoch, sample_rate=sample_rate)

            print(f'Epoch {epoch}/{num_epochs}, Training Loss: {train_avg_epoch_loss}')

        # # Validation loop
        # encoder.eval()
        # decoder.eval()
        # val_epoch_loss = 0
        # val_epoch_kl_div = 0
        # val_epoch_criterion = 0
        # with torch.no_grad():
        #     for batch, (dry_audio, wet_audio) in enumerate(val_loader):
        #         #reshape audio
        #         dry_audio = dry_audio[0:1, :]
        #         wet_audio = wet_audio[0:1, :]  
            
        #         dry_audio = dry_audio.view(1, 1, -1)
        #         wet_audio = wet_audio.view(1, 1, -1)
        #         wet_audio = wet_audio[:,:, :dry_audio.shape[-1]]
                

        #         dry_audio, wet_audio = dry_audio.to(device), wet_audio.to(device)

        #         # Pad both dry and wet audio to next power of 2
        #         dry_audio = center_pad_next_pow_2(dry_audio)
        #         wet_audio = center_pad_next_pow_2(wet_audio)
                
        #         # Apply PQMF to input
        #         dry_audio_decomposed = pqmf(dry_audio)
        #         wet_audio_decomposed = pqmf(wet_audio)

        #         audio_difference_decomposed = wet_audio_decomposed - dry_audio_decomposed
        #         audio_difference = wet_audio - dry_audio
    
        #         # Zero the parameter gradients
        #         optimizer.zero_grad()

        #         # Forward pass through encoder
        #         encoder_outputs = []
        #         x = dry_audio_decomposed
        #         for block in encoder.blocks:
        #             x = block(x)
        #             encoder_outputs.append(x)
        
        #         # Get the final encoder output
        #         z= encoder_outputs.pop()

        #         # Reverse the list of encoder outputs for the decoder
        #         encoder_outputs = encoder_outputs[::-1]
        #         encoder_outputs.append(dry_audio_decomposed)

        #         # Forward pass through encoder
        #         if use_kl:
        #             mu, logvar = encoder(dry_audio_decomposed)
        #             z = encoder.reparameterize(mu, logvar)
        #             kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/ mu.shape[-1]
        #             val_epoch_kl_div += kl_div

        #         # Forward pass through decoder
        #         net_outputs_decomposed = decoder(z, encoder_outputs)

        #         net_outputs = pqmf.inverse(net_outputs_decomposed)

        #         # # Trim outputs to original length
        #         original_length = dry_audio.shape[-1]
        #         net_outputs = net_outputs[..., :original_length]
        #         wet_audio = wet_audio[..., :original_length]
        #         dry_audio = dry_audio[..., :original_length]

        #         # Compute loss
        #         loss = criterion(net_outputs + dry_audio, wet_audio)
        #         if use_kl:
        #             loss += kl_div
            
        #         # Output
        #         output = net_outputs + dry_audio

        #         # Add KL divergence to the loss
        #         val_epoch_loss += loss 
                
        #         val_epoch_criterion += loss

        # val_avg_epoch_loss = val_epoch_loss / len(val_loader)
        # val_avg_epoch_loss_criterion = val_epoch_criterion / len(val_loader)  
        # if use_kl:
        #     val_avg_epoch_kl_div = val_epoch_kl_div / len(val_loader)

        # # Log loss
        # tensorboard_writer.add_scalar("Loss/ validation loss", val_avg_epoch_loss, epoch)
        # tensorboard_writer.add_scalar("Loss/ validation criterion", val_avg_epoch_loss_criterion, epoch)
        # if use_kl:
        #     tensorboard_writer.add_scalar("Loss/validation kl_div", val_avg_epoch_kl_div, epoch)

            print("Copying logs to host")
            os.system(f"rsync -rv --progress --inplace {tensorboard_writer.get_logdir()} {config.get_env_variable('TUSTU_TENSORBOARD_HOST')}:Data/{config.get_env_variable('TUSTU_PROJECT_NAME')}")

    tensorboard_writer.flush()

    print('Finished Training')
