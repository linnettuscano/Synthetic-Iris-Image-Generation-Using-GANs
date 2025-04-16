👁️ Synthetic Iris Image Generation using GANs
This project explores the use of Generative Adversarial Networks (GANs) to generate high-resolution synthetic iris images, with a focus on evaluating their potential to spoof biometric authentication systems.

📁 Project Overview
The notebook performs the following tasks:

Extracts and processes a dataset of iris images.

Defines a data generator for grayscale image batching and preprocessing.

Trains a GAN (using a custom train_gan function assumed to be defined elsewhere).

Evaluates both the discriminator and generator models using classification metrics.

Visualizes generated images.

Packages and downloads the output results.

🛠️ Technologies & Libraries Used
Python 3 (Colab)

NumPy

OpenCV (cv2)

Keras (assumed usage in GAN training)

Matplotlib

scikit-learn

Google Colab APIs (for file handling and download)

📂 Dataset
The dataset is stored in a ZIP file (dataa.zip) and contains grayscale iris images in multiple formats (.png, .jpg, etc.). The structure is recursively read to extract all valid image files.

🚀 How it Works
1. Extract Dataset
python
Copy
Edit
zipfile.ZipFile.extractall()
Unzips the dataset for further processing.

2. Data Generator Class
DataGenerator inherits from keras.utils.Sequence to:

Batch grayscale iris images

Resize them to 64x64

Normalize pixel values to [-1, 1]

3. GAN Training (Assumed Pre-defined Function)
python
Copy
Edit
train_gan(generator, discriminator, gan, data_generator, epochs=5000, batch_size=32, save_interval=500)
Trains the generator and discriminator on batches from the dataset.

4. Evaluation
Discriminator: Evaluated on real + fake images using classification_report.

Generator: Fake images are classified by the discriminator; performance is again evaluated using a confusion matrix-style report.

5. Visualization
The best generated images are displayed using matplotlib.

6. Output
Generated images are saved, zipped, and automatically downloaded via:

python
Copy
Edit
!zip -r generated_images.zip generated_images/
files.download("generated_images.zip")
📊 Evaluation Metrics
Accuracy, precision, recall, and F1-score using sklearn.metrics.classification_report

Separate reports are generated for the discriminator and generator.

🖼️ Sample Output
Visual outputs from the generator are grayscale iris images generated from random noise vectors.

📌 Note
This code assumes that the following models and methods are defined elsewhere:

generator: Keras/TensorFlow-based image generator

discriminator: Binary classifier for real vs fake images

gan: Combined model for training the generator

train_gan(...): Custom training loop for adversarial training

📥 Setup Instructions
Upload your dataset as a ZIP file (dataa.zip) to your Colab session.

Ensure the generator, discriminator, and GAN models are defined.

Run all cells to preprocess data, train the GAN, and evaluate results.
