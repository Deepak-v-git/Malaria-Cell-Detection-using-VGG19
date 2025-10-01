🦟 Malaria Cell Detection using VGG19 (Transfer Learning)

This project applies Transfer Learning with VGG19 pretrained on ImageNet to classify malaria cell images into infected (Parasitized) and uninfected categories. It demonstrates the use of deep learning for medical image analysis, specifically malaria detection, to support early and reliable diagnosis.

📊 Dataset

The dataset used is from Kaggle:
🔗 Malaria Cell Images Dataset

It consists of:

Parasitized cells (infected)

Uninfected cells (healthy)

To download:

kaggle datasets download -d iarunava/cell-images-for-detecting-malaria


Extract it into:

Train/  # Training set
Test/   # Validation/Test set

🏗️ Model Architecture

Base Model: VGG19 (pretrained on ImageNet, with frozen convolutional layers)

Custom Head: Flatten → Dense(2, softmax)

Input: 224×224 RGB images

Loss Function: Categorical Crossentropy

Optimizer: Adam

The model is trained with ImageDataGenerator for data augmentation (rescaling, zoom, shear, horizontal flips).

⚙️ Requirements

Install dependencies:

pip install -r requirements.txt


Main libraries:

Python 3.x

TensorFlow / Keras

NumPy

Matplotlib

OpenCV

scikit-learn

📈 Training Results

Epochs: 20

Training & validation accuracy plotted

Model saved as model_vgg19.h5

📌 Example results:

Training Accuracy: ~93.9%

Validation Accuracy: ~87.3%

📌 How to Run

Clone the repo:

git clone https://github.com/your-username/Malaria-Detection-VGG19.git
cd Malaria-Detection-VGG19


Download dataset from Kaggle and place in Train/ and Test/ folders.

Install dependencies:

pip install -r requirements.txt


Run Jupyter Notebook:

jupyter notebook Malaria-prediction.ipynb


To test on a single image:

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('model_vgg19.h5')
img = image.load_img('2.png', target_size=(224,224))
x = image.img_to_array(img) / 255
x = np.expand_dims(x, axis=0)

pred = np.argmax(model.predict(x), axis=1)
if pred[0] == 1:
    print("Uninfected")
else:
    print("Infected")

📸 Visualizations


<img width="547" height="413" alt="image" src="https://github.com/user-attachments/assets/5f0010db-03f2-46ad-b8f6-465d8c0b4235" />


📜 License

This project is open-source under the MIT License.
