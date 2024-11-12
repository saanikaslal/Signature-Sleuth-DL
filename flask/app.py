from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')  # Load your pre-trained model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if files are present
    if 'image1' not in request.files or 'image2' not in request.files:
        return "No file part", 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if file1.filename == '' or file2.filename == '':
        return "No selected file", 400

    image1 = Image.open(file1.stream).resize((112, 112))
    image2 = Image.open(file2.stream).resize((112, 112))

    image1 = img_to_array(image1)
    image2 = img_to_array(image2)

    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)

    image1 = image1.astype('float32')
    image2 = image2.astype('float32')

    # Perform inference
    prediction = model.predict([image1, image2])
    similarity_score = prediction[0][0]
    sign = 'Forged' if similarity_score > 0.5 else 'Genuine'

    return jsonify({'prediction': sign})

if __name__ == '__main__':
    app.run(debug=True)
