# GAN Projects: CycleGAN and Simple GAN

## Overview
This repository contains implementations of two Generative Adversarial Network (GAN) projects:
1. CycleGAN for image-to-image translation between Monet paintings and photographs.
2. Simple GAN for generating MNIST-like digit images.

## Project 1: CycleGAN (Monet to Photo)

### Dataset
- Monet2Photo dataset: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

### Tools and Libraries
- TensorFlow and Keras
- NumPy
- Matplotlib
- scikit-learn
- PIL (Python Imaging Library)
- Google Colab and Google Drive
- keras-contrib (for InstanceNormalization)

### Key Components
- Generator and Discriminator networks using convolutional layers
- Composite model for CycleGAN training
- Custom data loading and preprocessing
- Image pool for discriminator update

### Training Process
- Iterative training of generators and discriminators
- Multiple loss functions: adversarial, cycle consistency, and identity losses
- Periodic saving of model weights and generated images

## Project 2: Simple GAN (MNIST Generation)

### Dataset
- MNIST dataset (built-in Keras dataset)

### Tools and Libraries
- Keras
- NumPy
- Matplotlib
- Google Colab and Google Drive

### Key Components
- Generator: Fully connected layers with LeakyReLU and BatchNormalization
- Discriminator: Fully connected layers with LeakyReLU
- Adam optimizer

### Training Process
- Alternating training of discriminator and generator
- Binary cross-entropy loss
- Periodic saving of generated images

## What I Learned

### Deep Learning Concepts
- GAN architectures: CycleGAN and Simple GAN
- Balancing generator and discriminator training
- Importance of loss function design in GANs

### TensorFlow and Keras Skills
- Building complex model architectures
- Implementing custom training loops
- Using various layer types and activation functions

### Image Processing and Data Handling
- Working with different image datasets (Monet paintings, photographs, MNIST)
- Image preprocessing and augmentation techniques

### Project Management and Execution
- Organizing multiple deep learning projects
- Utilizing cloud resources for training
- Implementing logging and visualization for model progress

## Future Improvements
- Experiment with cross-application of techniques between projects
- Implement additional GAN variants (e.g., DCGAN, WGAN)
- Explore applications to higher resolution images or different domains
- Add quantitative evaluation metrics for generated images
- Optimize code for better performance and resource utilization

## Running the Projects

### CycleGAN
1. Mount Google Drive and set up the Monet2Photo dataset.
2. Run the provided notebook in Google Colab.
3. Monitor training progress and view generated images periodically.

### Simple GAN
1. Mount Google Drive for saving results.
2. Run the provided script in Google Colab.
3. Generated MNIST-like images will be saved at specified intervals.

## Conclusion
These projects demonstrate the implementation of two different GAN architectures: the complex CycleGAN for style transfer between domains, and a Simple GAN for generating digit images. They showcase a range of skills in deep learning, image processing, and GAN-specific techniques, providing a solid foundation for further exploration in generative models.

## References
- [Original CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- [Simple GAN Implementation by Aladdin Persson](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
