import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = MobileNetV2(weights='imagenet')

def predict_species(img):
    preds = model.predict(img)
    preds_numpy = preds if tf.executing_eagerly() else preds.numpy()
    return decode_predictions(preds_numpy, top=3)[0]

def retrain_model(corrected_data_dir='data/corrected', epochs=5):
    # Load the base model
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(os.listdir(corrected_data_dir)), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = datagen.flow_from_directory(corrected_data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

    model.fit(train_generator, epochs=epochs)
    model.save('model_retrained.h5')
