# Reviving Bach
Music generation using various approaches and neural networks

## Training Neural Networks

Files that begin with "Training" contains code that trains the neural networks. Text within the brackets specify the approach used, as well as the training scheme:

- AE: Vanilla autoencoder.
- VAE: Variational autoencoder.
- GAN: Generative adversarial network.
- joint: "Joint" training scheme.
- sep: "Separate" training scheme.
- samp: "Upsample" training scheme.
- mix: "Mixture" training scheme.

## Generating New Music

Files that begin with "Generation" reconstructs or generates music.

## Music Files

The folder *music* contains mp3 files that contains original, reconstructed or generated music. Some of the music files have random names! *Key.npy* maps the names back to the type of neural network that generated said file, as well as whether the music is reconstructed, or newly generated.
