from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
#import pdb
#from pprint import pprint
#########################################
#
#   This loads the model and asserts if its bullish or bearish
#
#########################################

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("model.h5")
print("Loaded model from disk")

# load model on test data
test_image=image.load_img('dataset/realtime_image/1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis =0)

# evaluate loaded model on test data
result = classifier.predict(test_image)
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
training_set.class_indices
#pdb.set_trace()
 
print(result[0][0])

if result[0][0] == 1:
    prediction = 'bull'
else:
    prediction = 'bear'
print(prediction)


# #plot the image for crosscheck
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# img=mpimg.imread('dataset/realtime_image/1.jpg')
# imgplot = plt.imshow(img)
# plt.show()