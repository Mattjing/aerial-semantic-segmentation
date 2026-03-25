
# import libraries
import os
import pandas as pd
import numpy as np

from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

import torch
from sklearn.model_selection import train_test_split


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout

import matplotlib.pyplot as plt



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


img_path = 'C:/Users/mattj/Documents/McMaster/MEST/SEP 769/archive/dataset/semantic_drone_dataset/original_images/'
mask_path = 'C:/Users/mattj/Documents/McMaster/MEST/SEP 769/archive/dataset/semantic_drone_dataset//label_images_semantic/'
names = list(map(lambda x: x.replace('.jpg', ''), os.listdir(img_path)))  ##his is using the map function to apply a lambda function to each element in the list of filenames returned by os.listdir(img_path). The lambda function takes a single argument x and returns x with '.jpg' replaced by an empty string. The result of the map function will be a new list containing the modified strings.


class_df = pd.read_csv('C:/Users/mattj/Documents/McMaster/MEST/SEP 769/archive/class_dict_seg.csv')
class_df


CLASSES = class_df.name.to_list()
CLASSES


n_classes = len(CLASSES)
n_classes


# Define a function to create a dataframe contains the images
def create_df():
    name = []
    for dirname, _, filenames in os.walk(img_path):
        for filename in filenames:
            name.append(filename.split('.')[0])

    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df_images = create_df()
print('Total Images: ', len(df_images))


# Show a sample image with its size information
img = Image.open(img_path + df_images['id'][100] + '.jpg')
mask = Image.open(mask_path + df_images['id'][100] + '.png')
print('Image Size', np.asarray(img).shape)
print('Mask Size', np.asarray(mask).shape)


plt.imshow(img)
plt.imshow(mask, alpha=0.6)
plt.title('Picture with Mask Appplied')
plt.show()


def multi_unet_model(n_classes=23, IMG_HEIGHT=4000, IMG_WIDTH=6000, IMG_CHANNELS=3):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.1)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)  # Original 0.1
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)  # Original 0.1
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])
    
    model.summary()
    
    return model


model = multi_unet_model()


def load_data(image_paths, mask_paths, img_height, img_width, n_classes):
    images = []
    masks = []
    for img_path, mask_path in zip(image_paths, mask_paths):
        image = load_img(img_path, target_size=(img_height, img_width))
        image = img_to_array(image)
        mask = load_img(mask_path, target_size=(img_height, img_width), color_mode="grayscale")
        mask = img_to_array(mask)
        mask = to_categorical(mask, num_classes=n_classes)
        images.append(image)
        masks.append(mask)
    return np.array(images), np.array(masks)

def train_unet(image_dir, mask_dir, num_epochs=25, batch_size=16, learning_rate=0.001, img_height=4000, img_width=6000, n_classes=23):
    # List all images and masks
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg') or fname.endswith('.png')])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')])
    
    # Split data into train+val and test sets
    img_trainval, img_test, mask_trainval, mask_test = train_test_split(image_paths, mask_paths, test_size=0.1, random_state=19)

    # Split train+val set into train and val sets
    img_train, img_val, mask_train, mask_val = train_test_split(img_trainval, mask_trainval, test_size=0.2, random_state=19)

    # Load data
    X_train, y_train = load_data(img_train, mask_train, img_height, img_width, n_classes)
    X_val, y_val = load_data(img_val, mask_val, img_height, img_width, n_classes)
    X_test, y_test = load_data(img_test, mask_test, img_height, img_width, n_classes)

    # Initialize the model
    model = multi_unet_model(n_classes=n_classes, IMG_HEIGHT=img_height, IMG_WIDTH=img_width, IMG_CHANNELS=3)

    # Train the model
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        batch_size=batch_size, 
                        epochs=num_epochs)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    return model, history


image_dir = img_path
mask_dir = mask_path
trained_model, training_history = train_unet(image_dir, mask_dir)


plot_acc(history)


plot_score(history)


def plot_predictions(model, X_test, y_test, num_images=5):
    preds = model.predict(X_test)
    plt.figure(figsize=(15, 5 * num_images))

    for i in range(num_images):
        index = np.random.randint(0, len(X_test))

        plt.subplot(num_images, 3, i * 3 + 1)
        plt.title("Original Image")
        plt.imshow(X_test[index].astype('uint8'))

        plt.subplot(num_images, 3, i * 3 + 2)
        plt.title("True Mask")
        plt.imshow(np.argmax(y_test[index], axis=-1))

        plt.subplot(num_images, 3, i * 3 + 3)
        plt.title("Predicted Mask")
        plt.imshow(np.argmax(preds[index], axis=-1))

    plt.show()



plot_predictions(model, X_test, y_test)
