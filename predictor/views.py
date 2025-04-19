import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Load model once at start
import tensorflow as tf

# Load model only once, not on every request
model = tf.keras.models.load_model('CNN_model.h5', compile=False)
def load_and_preprocess_image(image_path, target_size=(50, 50)):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size)
        img_array = img_resized / 255.0
        return img_resized, img_array
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def index(request):
    return render(request, 'predictor/index.html')

def predict(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        fs = FileSystemStorage()
        filepath = fs.save(f"uploaded_images/{image.name}", image)
        full_path = fs.path(filepath)

        img_for_display, img_array = load_and_preprocess_image(full_path)
        if img_array is None:
            return render(request, 'predictor/result.html', {'error': 'Invalid image!'})

        img_batch = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_batch)
        cancer_prob = float(prediction[0][0])
        predicted_class = 'Cancer' if cancer_prob >= 0.5 else 'Normal'

        return render(request, 'predictor/result.html', {
            'image_url': fs.url(filepath),
            'predicted_class': predicted_class,
            'cancer_prob': f"{cancer_prob*100:.2f}%"
        })
    return render(request, 'predictor/index.html')
