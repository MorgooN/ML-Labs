import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten
from keras.datasets import mnist
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
import numpy as np


# load image
file = r'D:\Git\ML-Labs\LAB_2_Images\dataTraining.png'
test_im = cv2.imread(file,cv2.IMREAD_GRAYSCALE)

# Format image
img_resized = cv2.resize(test_im, (28,28), interpolation=cv2.INTER_LINEAR)
img_resized = cv2.bitwise_not(img_resized)
img_resized = tf.constant(img_resized, dtype=tf.float32) / 255

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# minmax нормализация 
train_images = tf.constant(train_images, dtype=tf.float32) / 255
test_images = tf.constant(test_images, dtype=tf.float32) / 255

 
# Преобразование меток классов в one-hot кодировку
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def createModel(input_shape):
    # NN creation. 1 hidden layer, 88 neurons

    input_layer = Input(shape = input_shape)
    flatten_layer = Flatten()(input_layer) # flatten layer is used to transform imagers (matrix) into the vectors
    hidden_1 = Dense(units=88,activation='relu')(flatten_layer) # create layer with 88 neurons and ReLU-func
    output = Dense(units = 10, activation='softmax')(hidden_1) # output layer with 10 neuron and softman-func
    return Model(inputs = input_layer, outputs = output)

def trainModel(X_train, Y_train):
    save_callbacks = ModelCheckpoint(filepath='best_model.h5',
                                     monitor='val_loss',
                                     save_best_only=True,
                                     save_weights_only=True,
                                     verbose=1)
    

    model = createModel((28,28))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
              metrics = 'accuracy')

    model.fit(x = X_train,
          y = Y_train,
          epochs = 30,
          batch_size = 64,
          validation_split = 0.1,
          callbacks=[save_callbacks])
    return model

model = trainModel(train_images, train_labels)

# graphichs for accuracy
plt.figure()
plt.plot(model.history.history["accuracy"], label="training accuracy")
plt.plot(model.history.history["val_accuracy"], label="validation accuracy")
plt.legend()
plt.show()

# graphics for value loss
plt.figure()
plt.plot(model.history.history["loss"], label="training loss")
plt.plot(model.history.history["val_loss"], label="validation loss")
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")


sample_index = 25 # берём индекс конкретного примера из обучающего набора
X_sample, y_sample = test_images[sample_index], test_labels[sample_index] # достаём из тестовго набора картинку и метку для него
y_pred = model.predict(img_resized[None, :]) # делаем предсказание для данного примера.
plt.imshow(img_resized) # рисуем картинку
plt.show()

np.set_printoptions(suppress=True) # чтобы вывод предсказаний y_pred был в понятном формате

print(f"Предсказанные вероятности нашей сети: {y_pred}")
print(f"Итоговый ответ: {np.argmax(y_pred)}") # argmax осуществляет нахождения индекса (позиции) максимального элемента в массиве
print(f"Правильный ответ: {y_sample}")
