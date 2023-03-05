# pytorch_example_project

Sample project made for the purpose of learning to use pytorch.

Model architectures can be found in their respective files ([cnn.py](https://github.com/AlaNeponen/pytorch_example_project/blob/main/cnn.py) and [autoencoder.py](https://github.com/AlaNeponen/pytorch_example_project/blob/main/autoencoder.py))

Program runs as follows:

1. Train a CNN (model_1) to classify images from MNIST dataset
2. Train an autoencoder (model_2) to reconstruct images from MNIST dataset
3. Test how well model_1 can classify the reconstructed images produced by model_2

A shortened version of the output of the program is found in [output.txt](https://github.com/AlaNeponen/pytorch_example_project/blob/main/output.txt).

A figure comparing the original test images (top) to the reconstructed ones (bottom) is [found here](https://github.com/AlaNeponen/pytorch_example_project/blob/main/Figure_3.png).
