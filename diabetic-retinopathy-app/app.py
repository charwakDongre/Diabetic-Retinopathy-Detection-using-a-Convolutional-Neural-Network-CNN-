import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24) # Necessary for flash messages

# Configuration
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/re_model.h5"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
try:
    model = load_model(MODEL_PATH)
except (IOError, ImportError):
    model = None # Handle case where model is not found

# Class labels
classes = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if model is None:
        flash("Model not found. Please train the model first.", "error")
        return redirect(url_for("home"))
        
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            flash("No file part in the request.", "warning")
            return redirect(url_for("home"))

        file = request.files["file"]

        # If the user does not select a file, the browser submits an empty file
        if file.filename == "":
            flash("No file selected. Please choose an image.", "warning")
            return redirect(url_for("home"))

        # Check if the file type is allowed
        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload a .png, .jpg, or .jpeg file.", "warning")
            return redirect(url_for("home"))

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        # Pass results to the template
        return render_template(
            "results.html",
            diagnosis=classes.get(predicted_class, "Unknown Class"),
            confidence=f"{confidence:.2f}%",
            original_image=filename
        )
    
    # If GET request, just show the home page
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)