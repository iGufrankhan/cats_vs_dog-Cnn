from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import glob

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = os.getenv("MODEL_PATH", "model/dog_cat_cnn.h5")
IMG_SIZE = (150, 150)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def clear_upload_folder():
    """Remove all files from upload folder"""
    files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error removing {f}: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if model is None:
            return render_template("index.html", error="Model not loaded. Please check configuration.")
        
        # Clear old uploads
        clear_upload_folder()
        
        file = request.files["image"]
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=IMG_SIZE)
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]
        result = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

        # Return correct image URL
        image_url = url_for('static', filename=f'uploads/{filename}')
        return render_template("index.html", result=result, image=image_url)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
