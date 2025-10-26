# Celebrity Image Generator with DCGAN
This project is an implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) using Keras and TensorFlow. The model is trained on the CelebA (Celebrity Faces Attributes) dataset to generate new, realistic 128x128 images of celebrity faces. You can view the dataset here https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

## 1. Data Loading & Preprocessing 🧹
The initial phase involves loading and preparing the CelebA dataset for training the GAN.

Loading Data: Images are loaded from the img_align_celeba directory.

Image Cropping: The original images (218x178) are cropped to a square format (178x178) to center the face.

Resizing: All cropped images are resized to 128x128 to provide a uniform input size for the model.

Normalization: Pixel values are normalized from the [0, 255] range to the [-1, 1] range. This is a crucial step to match the generator's output tanh activation function.

Data Visualization: A sample of the processed images is plotted to verify the preprocessing steps.

## 2. The Model: DCGAN Architecture ⚙️
A Generative Adversarial Network consists of two competing neural networks: a Generator and a Discriminator.

Generator
The Generator's job is to create realistic images from random noise.

Input: A 100-dimension latent vector (random noise).

Architecture: It uses a Dense layer to project the noise, followed by a series of Conv2DTranspose layers to upsample the data from a low-resolution feature map (16x16) up to the final 128x128x3 image.

Activations: LeakyReLU is used for intermediate layers, and a tanh activation is used on the final layer to produce the [-1, 1] normalized image.

Discriminator
The Discriminator's job is to act as a binary classifier, determining whether an input image is real (from the dataset) or fake (from the generator).

Input: A 128x128x3 image.

Architecture: It is a standard CNN composed of Conv2D layers with strides to downsample the image.

Activations: LeakyReLU is used for all convolutional layers. A Dropout layer is included for regularization.

Output: A single Dense node with a sigmoid activation, outputting a probability between 0 (fake) and 1 (real).

## 3. Training Process 🧑‍💻
The two models are trained in an adversarial manner:

Train Discriminator: The discriminator is trained on a combined batch of real images (labeled as 1) and fake images (labeled as 0). Its weights are updated to get better at telling them apart.

Train Generator: The generator is trained by feeding its output to the discriminator. For this step, the discriminator's weights are frozen, and the generator's output is mislabeled as 1 (real). The generator's weights are then updated to "fool" the discriminator more effectively.

This process is repeated for 50 epochs, with the models progressively improving. The optimizer used for both models is Adam with a learning rate of 0.0002 and beta_1 of 0.5, which are standard for GANs.

## 4. Generating Results & Saving Model ⚒️
To monitor progress, a fixed set of noise vectors (seed) is used to generate a sample of images every 5 epochs. This allows for a clear visualization of how the generator improves over time.

After training is complete, the generator's weights are saved to a file (generator_epoch_50.weights.h5) and a jpg file that you can view. This file contains the trained model and can be loaded later to generate new faces without retraining.

