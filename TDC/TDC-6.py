import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import Adam
data = tfds.load("iris", split=tfds.split.TRAIN.subsplit(tfds.percent[:80]))

def preprocess(features):
    return features["features"], tf.one_hot(features["label"], depth=3)

def solution_model():
    train_dataset = data.map(preprocess).batch(10)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(train_dataset, epochs=100)
    return model




if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
