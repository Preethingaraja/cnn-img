import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
training_datagram = ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set=training_datagram.flow_from_directory(r"C:\Users\habit\Desktop\CNN\training_set",target_size=(64,64),batch_size=32,class_mode='binary')
test_dataget=ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_set=test_dataget.flow_from_directory(r"C:\Users\habit\Desktop\CNN\test_set",target_size=(64,64),batch_size=32,class_mode='binary')
cnn=tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3 ,activation='relu',input_shape=(64,64,3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=(64,64,3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
cnn.fit(training_set,validation_data=test_set,epochs=10)
import numpy as np 
from keras.preprocessing import image
test_image=tf.keras.utils.load_img(r"C:\Users\habit\Desktop\CNN\test_set\cats\cat.14.jpg",target_size=(64,64))
test_img=tf.keras.utils.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image)
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'
print(prediction)
 
import matplotlib.pyplot as plt

history=cnn.fit(training_set,validation_data=test_set,epochs=10)
test_loss,test_acc=cnn.evaluate(test_set)
print('Test Loss:',test_loss)
print('Test Accuarcy',test_acc)
plt.figure(dpi=300)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','val'],loc='best')
plt.show()
plt.figure(dpi=300)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train','val'],loc='upper left')
plt.show()