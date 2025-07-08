import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam',
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5)

# model.save('handwritten_model.h5')

model = tf.keras.models.load_model('handwritten_model.h5')
loss, acc = model.evaluate(x_test, y_test)

print("Model accuracy: ", acc)
print("Model loss: ", loss)

img_num = 1
while os.path.isfile(f"TestSet/num{img_num}.png"):
    try:
        img_path = f"TestSet/num{img_num}.png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to 28x28
        img = cv2.resize(img, (28, 28))

        # ** Invert colors if background is black
        if np.mean(img) > 127:
            img = cv2.bitwise_not(img)

        # Normalize and reshape
        img = img / 255.0
        img = img.reshape(1, 28, 28)

        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(f"Predicted: {np.argmax(prediction)}")
        plt.axis('off')
        plt.show()
        # img = cv2.imread(f"TestSet/num{img_num}.png")[:,:,0]
        # img = np.invert(np.array([img]))
        # prediction = model.predict(img)
        # print(f"This digit is probably a {np.argmax(prediction)}")
        # plt.imshow(img[0], cmap=plt.cm.binary)
        # plt.show()
    except Exception as e:
        print(f"Error processing image {img_num}: {e}")
    finally:
        img_num += 1

# For white on black background images, you might need to invert the colors which is exactly what we do above. We do it HERE **