#Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from cv2 import cv2


#Initializing CNN
classifier = Sequential()

#1Convolution
classifier.add(Convolution2D(32,3,3,input_shape = (224,224,3), activation = 'relu'))

#2Pooling
classifier.add(AveragePooling2D(pool_size=(4, 4)))

#adding 2nd 3rd and 4th convolution layer
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
classifier.add(AveragePooling2D(pool_size=(4, 4)))


#3Flattening
classifier.add(Flatten())

#4Full_Connection

classifier.add(Dense(output_dim=32,activation = 'relu'))

classifier.add(Dense(output_dim=64,activation = 'relu'))

classifier.add(Dense(output_dim=128,activation = 'relu'))
classifier.add(Dense(output_dim=256,activation = 'relu'))
classifier.add(Dense(output_dim=256,activation = 'relu'))

classifier.add(Dense(output_dim=1,activation = 'sigmoid'))

#Compiling CNN
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

#Fitting CNN to images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
print("\nTraining the data...\n")
training_set = train_datagen.flow_from_directory('train',
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('test',
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         samples_per_epoch=420,
                         nb_epoch = 1,
                         validation_data =test_set,
                         nb_val_samples = 25)

#Making new predictions
print("\nMaking predictions for uploaded X-ray...\n")
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('images//nocorona.jpg',target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
(covid,nocovid) = classifier.predict(test_image)[0]
prob = covid if covid > nocovid else nocovid
print(prob*100)
training_set.class_indices
if result[0][0] == 0:

    prediction = 'Patient is affected with Corona'
else:
    prediction = 'Patient is Healthy'

print("\nOutcome : ",prediction)
print(prob*100)
 


