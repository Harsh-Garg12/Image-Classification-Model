import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json


dict = {
    0: 'Building',
    1: 'Forest',
    2: 'Glacier',
    3: 'Mountain',
    4: 'Sea',
    5: 'Street'
}


def image_label(image):
    # load json and create model
    json_file = open('model1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model1 = model_from_json(loaded_model_json)
    # load weights into new model
    model1.load_weights("model1.h5")
    print("Loaded model from disk")

    model1.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    L = model1.predict(image)

    for i in range(len(L[0])):
        if abs(1. - L[0][i]) <= 1e-5:
            return dict[i]

