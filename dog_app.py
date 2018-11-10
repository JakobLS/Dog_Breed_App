# dog_app

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import random
import cv2                
import matplotlib.pyplot as plt                        
#matplotlib inline
import time
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import ImageFile 



""" define function to load train, test, and validation datasets """
# keras.utils.to_categorical(y, num_classes=None)
# Converts a class vector (integers) to binary class matrix.
# y: class vector to be converted into a matrix (integers from 0 to num_classes).
# num_classes: total number of classes.
# Returns a binary matrix representation of the input.
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are {0} total dog categories.'.format(len(dog_names)))
#print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
#print('There are {0} total dog images.\n'.format(len(np.hstack([train_files, valid_files, test_files]))))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

#print(dog_names)


"""Import human dataset"""
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))      # /*/* = Load everything that's in the lfw folder
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))



"""Detect humans """
###### extract pre-trained face detector ######
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[2])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


"""Write a face detector """
# returns "True" if face is detected by the face_cascade function in image stored at img_path.
def face_detector(img_path, face_cascade):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1)
    return len(faces) > 0


"""Assess the human face detector"""
human_files_short = human_files[:100]
dog_files_short = train_files[:100]

# Take the time
start = time.time()

def perfTest(a_face_detector, face_cascade):
    ## TODO: Test the performance of the face_detector algorithm 
    ## on the images in human_files_short and dog_files_short.
    human_detected_as_human = 0
    human_detected_as_dog = 0
    for i in range(100):
        if a_face_detector(human_files_short[i], face_cascade):
            human_detected_as_human += 1
        elif a_face_detector(dog_files_short[i], face_cascade):
            human_detected_as_dog += 1
            
    print("Percentage of human faces detected among the first 100 human pictures: {0}% "
          .format(human_detected_as_human/len(human_files_short)*100))
    print("Percentage of human faces detected among the first 100 dog pictures: {0}% "
          .format(human_detected_as_dog/len(dog_files_short)*100))

#Print the performance   
perfTest(face_detector, face_cascade)

# Total time
end = time.time()
print("Total time taken to detect the faces: {0:0.2f}s".format(end-start))



"""Detect Dogs """

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    # (The fourth axis (from 3D to 4D) is added at position axis=0)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


"""Assess the dog detector"""
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

start = time.time()

human_detected_as_dog = 0
dog_detected_as_dog = 0
for i in range(len(human_files_short)):
    if dog_detector(human_files_short[i]):
        human_detected_as_dog += 1
    elif dog_detector(dog_files_short[i]):
        dog_detected_as_dog += 1

print("Percentage of humans detected as dogs: {0}%".format(human_detected_as_dog*100/len(human_files_short)))
print("Percentage of dogs detected as dogs: {0}%".format(dog_detected_as_dog*100/len(dog_files_short)))  


# Total time
end = time.time()
print("Total time taken to determine the performance: {0:0.2f}s".format(end-start))
           

"""Preprocessing the data """                
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


"""Create the CNN structure"""

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 

#
#model = Sequential()
#
#### TODO: Define your architecture.
#model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(224, 224, 3)))
#model.add(MaxPooling2D(pool_size=2))
#model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')) 
#model.add(MaxPooling2D(pool_size=2))
#model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
#model.add(MaxPooling2D(pool_size=2))
#model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
#model.add(MaxPooling2D(pool_size=2))
#model.add(GlobalAveragePooling2D())
#model.add(Dense(133, activation='softmax'))   # - 133 different breeds. This is a fully connected layer. 
#                                              # One can try to add several to increase the accuracy.
#                                              # - Adding Dropouts to curb overfitting
#                                              # - Stagging multiple convolutional layers before a single max pool
#                                              # might increase the accuracy as well. 
#        
#model.summary()
#
#
#"""Compile the model"""
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
#
#"""Train the model"""
#from keras.callbacks import ModelCheckpoint  
#import time
#
#### TODO: specify the number of epochs that you would like to use to train the model.
#epochs = 10
#
## Time the training
#start = time.time()
#
#checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
#                               verbose=1, save_best_only=True)
#
#model.fit(train_tensors, train_targets, 
#          validation_data=(valid_tensors, valid_targets),
#          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
#
##batch_size=20
##model.fit_generator(datagen_train.flow(train_tensors, train_targets, batch_size=batch_size),
##                    steps_per_epoch=train_tensors.shape[0] // batch_size,
##                    epochs=epochs, verbose=2, callbacks=[checkpointer],
##                    validation_data=datagen_valid.flow(valid_tensors, valid_targets, batch_size=batch_size),
##                    validation_steps=valid_tensors.shape[0] // batch_size)
#
#
#
## Print out how long it took to train
#end = time.time()
#tt = end-start
#print()
#print("Total training time was {0:0.0f} min and {1:0.0f}s. ".format((tt-tt%60)/60, tt%60))
#
#
#"""Load the model with the best validation Loss"""
#model.load_weights('saved_models/weights.best.from_scratch.hdf5')
#
#
#"""Test the model"""
## get index of predicted dog breed for each image in test set
#dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
#
## report test accuracy
#test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
#print('Test accuracy: %.4f%%' % test_accuracy)


"""Create a CNN to Classify Dog Breeds (using Transfer Learning)"""

### Obtain Bottleneck features from another pre-trained CNN

bottleneck_features = np.load('bottleneck_features/DogXceptionData.npz')
train_Xception = bottleneck_features['train']
valid_Xception = bottleneck_features['valid']
test_Xception = bottleneck_features['test']

print(test_Xception.shape)


""" Augment the traiing data """
#from keras.preprocessing.image import ImageDataGenerator

#train_Xception_to_use = train_Xception
#valid_Xception_to_use = valid_Xception

# create and configure augmented image generator
#datagen_train = ImageDataGenerator(
#    rotation_range = 40,
#    width_shift_range=0.2,  # randomly shift images horizontally (20% of total width)
#    height_shift_range=0.2,  # randomly shift images vertically (20% of total height)
#    horizontal_flip=True) # randomly flip images horizontally

# create and configure augmented image generator
#datagen_valid = ImageDataGenerator(
#    width_shift_range=0.2,  # randomly shift images horizontally (20% of total width)
#    height_shift_range=0.2,  # randomly shift images vertically (20% of total height)
#    horizontal_flip=True) # randomly flip images horizontally

# fit augmented image generator on data
#datagen_train.fit(train_Xception_to_use)
#datagen_valid.fit(valid_Xception_to_use)


"""Define CNN architecture"""
Xception_model = Sequential()

#Xception_model.add(Flatten(input_shape=train_Xception.shape[1:]))
#Xception_model.add(Dense(64, activation='relu'))
#Xception_model.add(Dropout(0.5))
#Xception_model.add(Dense(133, activation='softmax'))

#Xception_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:])) # (7, 7, 512)

Xception_model.add(GlobalMaxPooling2D(input_shape=train_Xception.shape[1:]))
Xception_model.add(Dropout(0.5))
Xception_model.add(Dense(133, activation='softmax'))

Xception_model.summary()


"""Compile the model"""
from keras import optimizers

#Xception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
Xception_model.compile(loss='categorical_crossentropy', 
                       optimizer = optimizers.SGD(lr=1e-4, momentum=0.9), 
                       metrics=['accuracy'])


"""Train the model"""
# Time the training
start = time.time()

### TODO: Train the model.
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5', 
                               verbose=1, save_best_only=True)

Xception_model.fit(train_Xception, train_targets, 
          validation_data=(valid_Xception, valid_targets),
          epochs=100, batch_size=20, callbacks=[checkpointer], verbose=1)


# Print how long it took
end = time.time()
tt = end-start
print("\nTotal training time was {0:0.0f} min and {1:0.0f} s. ".format((tt-tt%60)/60, tt%60))


"""Load the model with the best validation loss"""
### Load the model weights with the best validation loss.
Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')

"""Test the model"""
### TODO: Calculate classification accuracy on the test dataset.

# get index of predicted dog breed for each image in test set
Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) 
                           for feature in test_Xception]

# report test accuracy
test_accuracy = (100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/
                 len(Xception_predictions))
print('Test accuracy: %.4f%%' % test_accuracy)


"""Return the dog breed function"""
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

def Xception_predict_breed(img_path):
    # extract bottleneck features
    Xception_bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    
    # get the predicted vector
    predicted_vector = Xception_model.predict(Xception_bottleneck_feature)
    
    # return the predicted dog breed
    return dog_names[np.argmax(predicted_vector)]


"""Run the dog app"""

### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.
# I have tried to display the predicted dog breed as well but without success. =(

def import_files():
    files_to_check = np.array(glob("images/check_this/*"))   # Make sure that the images are JPEG! 
    #files_to_check = np.array(glob("dogImages/valid/107.Norfolk_terrier"))
    #print(files_to_check)
    return files_to_check                                    # if not, the kernel will stop running.

def get_dog_breed_picture(breed):
    length = len(breed) 
    for item in glob("dogImages/valid/*/*"):
        print(item[-(length + 10): -10])
        if item[-(length + 10): -10] == breed:
            print(item)
            breed_pics = np.array(glob(item))
            breed_pic = cv2.imread(item)
            cv_rgb2 = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            return cv_rgb2


def image_detector(file):
    # start by displaying the picture
    image = cv2.imread(file)
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    plt.figure()
    #plt.figure(figsize=(10, 10))
    #plt.subplot(2,2,1)
    plt.imshow(cv_rgb)
    plt.show()
    
    # return the breed
    breed = Xception_predict_breed(file)  
    
    # check whether the picture is a dog, human or neither
    if dog_detector(file):
        print("Dog breed: {0}".format(breed))
    elif face_detector(file, face_cascade):
        #dog_breed_pucture = get_dog_breed_picture(breed)
        #plt.subplot(2,2,2)
        #plt.imshow(dog_breed_pucture)
        print("Hey human!")
        print("You look like the dog breed: {0}".format(breed))
        print("Don't worry however, it's cute!")   
    else:
        print("It seems that I can't classify this picture.")

    
        
start = time.time()
  
    
images = import_files()
image_detector(images[0])  

end = time.time()
tt = end-start
print("\nTime to run the algorithm: {0:0.0f} min and {1:0.0f} s ".format((tt-tt%60)/60, tt%60))



## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

for i in range(len(images)):
    image_detector(images[i])
    print("="*80)



