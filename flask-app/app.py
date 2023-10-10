from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = load_model('mnist_cnn.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        # Get the uploaded image file from the request
        file = request.files['image']

        if file:
            # Read the image and preprocess it
            img = Image.open(io.BytesIO(file.read()))
            img = img.resize((28, 28))
            img = np.array(img.convert('L'))  # Convert to grayscale
            img = img.reshape(1, 28, 28, 1)
            img = img.astype('float32') / 255.0

            # Make predictions
            predictions = model.predict(img)
            result = np.argmax(predictions[0])

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
