### Build Classification Model to predict there is a ship on image or no
import keras
from keras.models import Sequential

def classification_model():
    model = Sequential()
    model.add(keras.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256,256,3)))
    model.add(keras.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.Dropout(0.2))

    model.add(keras.Flatten())

    model.add(keras.Dense(128, activation='relu'))
    model.add(keras.Dense(2, activation='sigmoid'))