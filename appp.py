from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)

# Model Yükleme
model = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)), 
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), 
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)), 
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)), 
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(2, activation='sigmoid') 
])

weights_path = r'C:\Users\doguk\OneDrive\Masaüstü\Breast Cancer2\model.weights.h5'  # Ağırlık dosyası yolu
if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {weights_path}")

# Yüklenen dosyaların saklanacağı klasör
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ana sayfa ve tahmin işlemi
@app.route("/", methods=["GET", "POST"])
def index():
    filename, stage, confidence = None, None, None  # Başlangıçta boş değerler

    if request.method == "POST":
        # Dosya kontrolü
        if "file" not in request.files:
            return render_template("index.html", filename=None, stage=None, confidence=None)
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", filename=None, stage=None, confidence=None)
        if file:
            # Dosyayı kaydetme
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Model ile tahmin yap
            stage, confidence = predict(filepath)

            # Sonuçları result.html'e iletmek için yönlendirme yap
            return redirect(url_for('result', stage=stage, confidence=confidence, filename=filename))

    # En son yüklenen dosyanın yolunu al
    last_uploaded_file = None
    if os.listdir(app.config["UPLOAD_FOLDER"]):
        last_uploaded_file = max(
            [os.path.join(app.config["UPLOAD_FOLDER"], f) for f in os.listdir(app.config["UPLOAD_FOLDER"])],
            key=os.path.getctime
        )
        last_uploaded_file = os.path.basename(last_uploaded_file)

    return render_template("index.html", filename=filename, stage=stage, confidence=confidence, last_uploaded_file=f"uploads/{last_uploaded_file}" if last_uploaded_file else None)

# Tahmin Fonksiyonu
def predict(filepath):
    # Görüntüyü yükleme ve ön işleme
    input_image = cv2.imread(filepath)
    input_image_resized = cv2.resize(input_image, (128, 128))
    input_image_scaled = input_image_resized / 255.0
    input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

    # Model tahmini
    input_prediction = model.predict(input_image_reshaped)
    input_pred_label = np.argmax(input_prediction)

    # Tahmin sonucu ve güven oranı
    result = "Kişide kanser bulunmamaktadır." if input_pred_label == 1 else "Kişide kanser bulunmaktadır."
    confidence = np.max(input_prediction) * 100

    return result, round(confidence, 2)

# Result route: Sonuçları görüntüle
@app.route("/result")
def result():
    stage = request.args.get("stage")
    confidence = request.args.get("confidence")
    filename = request.args.get("filename")
    file_url = f"{app.config['UPLOAD_FOLDER']}/{filename}" if filename else None
    return render_template("result.html", stage=stage, confidence=confidence, filename=filename, file_url=file_url)


if __name__ == "__main__":
    app.run(debug=True)
