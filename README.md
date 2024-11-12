<h1>Siamese Network for Similarity Learning</h1>
<p>This project implements a Siamese Neural Network using TensorFlow and Keras. The Siamese Network is designed to learn similarity between pairs of images, enabling tasks such as image matching, verification, and similarity-based ranking. This model uses convolutional layers to extract features from input images and a Manhattan Distance metric to measure similarity.</p>

<h1>Model Architecture</h1>
The Siamese Network model is composed of:

Convolutional Layers: Extract features from input images with increasing depth.
Pooling Layers: Reduce the spatial dimensions to manage computational load.
Dropout Layers: Prevent overfitting by randomly disabling neurons during training.
Flatten and Dense Layers: Flatten feature maps and generate embeddings.
Lambda Layer (Manhattan Distance): Calculates the Manhattan Distance between the embeddings of the two inputs.
Output Layer: A single neuron with sigmoid activation, outputting a probability score indicating whether the input images are similar.

<h1>License</h1>
This project is licensed under the MIT License.
