from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

model = MobileNetV2(weights='imagenet')

def predict_species(img):
    preds = model.predict(img)
    return decode_predictions(preds, top=3)[0]
