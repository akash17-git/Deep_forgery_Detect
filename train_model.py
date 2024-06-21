import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageChops, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

np.random.seed(2)
sns.set(style='white', context='notebook', palette='deep')

# Install and authenticate Kaggle API
os.system('pip install kaggle')
os.makedirs('/root/.kaggle/', exist_ok=True)
os.system('cp /path/to/kaggle.json /root/.kaggle/')  # Replace with the path to your kaggle.json file
os.system('chmod 600 /root/.kaggle/kaggle.json')

# Download and unzip dataset
os.system('kaggle datasets download -d divg07/casia-20-image-tampering-detection-dataset')
os.system('unzip casia-20-image-tampering-detection-dataset.zip -d /content/CASIA2')

def convert_to_ela_image(image_path, quality):
    original_image = Image.open(image_path)
    original_image.save('temp.jpg', 'JPEG', quality=quality)
    temporary_image = Image.open('temp.jpg')
    ela_image = ImageChops.difference(original_image, temporary_image)
    ela_image = ela_image.convert('L')
    ela_image = ImageEnhance.Brightness(ela_image).enhance(30)
    return ela_image

dataset = pd.read_csv('/content/CASIA2/train_label.csv')  # Adjust the path accordingly

X = []
Y = []

for index, row in dataset.iterrows():
    image_path = f"/content/CASIA2/{row['image_path']}"
    X.append(np.array(convert_to_ela_image(image_path, 90).resize((128, 128))).flatten() / 255.0)
    Y.append(row['label'])

X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 128, 128, 1)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(128, 128, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

optimizer = RMSprop(learning_rate=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, verbose=0, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

epochs = 30
batch_size = 100
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), verbose=2, callbacks=[early_stopping, reduce_lr])

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label='Training loss')
ax[0].plot(history.history['val_loss'], color='r', label='Validation loss')
ax[0].legend(loc='best', shadow=True)
ax[1].plot(history.history['accuracy'], color='b', label='Training accuracy')
ax[1].plot(history.history['val_accuracy'], color='r', label='Validation accuracy')
ax[1].legend(loc='best', shadow=True)
plt.show()

model.save('tamper_detection_model.h5')

Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(confusion_mtx, classes=range(2))
plt.show()
