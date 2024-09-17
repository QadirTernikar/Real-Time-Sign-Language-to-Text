from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
model = load_model('sign_language_model.h5')

# Helper function to process the image and predict
def predict_image(image):
    image = cv2.resize(image, (28, 28))  # Resize image
    image = image / 255.0  # Normalize image
    image = np.reshape(image, (1, 28, 28, 1))
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Route to render homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file'].read()
    image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_GRAYSCALE)

    prediction = predict_image(image)

    return jsonify({"prediction": chr(65 + prediction)})  # Return letter (e.g., 'A' -> 65 in ASCII)

if __name__ == "__main__":
    app.run(debug=True)
