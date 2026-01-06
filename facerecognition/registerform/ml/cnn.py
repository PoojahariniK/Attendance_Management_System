import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import cv2
import re
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
# Data loading and preprocessing
data_dir = pathlib.Path("c:/facerecog/facerecognition/media")
file_paths = [file for file in data_dir.rglob('*.jpg') if file.is_file()]
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
x = []
y = []
idv = []
for file in file_paths:
    val = re.findall(r"\d{10}", str(file))
    idv.extend(val)
idv = list(dict.fromkeys(idv))
dict_found = {ele: ind for ind, ele in enumerate(idv)}
print(dict_found)
num_class = len(dict_found)
for file in file_paths:
    img = cv2.imread(str(file))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    if len(faces)>0:
        (x_coord,y_coord,w,h)=faces[0]
        face_img=img[y_coord:y_coord+h,x_coord:x_coord+w]
    resi = cv2.resize(face_img, (224, 224))
    resi=cv2.cvtColor(resi,cv2.COLOR_BGR2RGB)
    #plt.imshow(resi)
    #plt.show()
    id = re.findall(r"\d{10}", str(file))
    id1 = id[0]
    x.append(resi)
    y.append(dict_found[id1])
x = np.array(x)
y = np.array(y)
resample=RandomUnderSampler(random_state=45)
x_resampled, y_resampled = resample.fit_resample(x.reshape((x.shape[0], -1)), y)
x_resampled = x_resampled.reshape(x_resampled.shape[0], 224, 224, 3)
xtrain, xtest, ytrain, ytest = train_test_split(x_resampled,y_resampled, test_size=0.1, random_state=100)
xtrain_scaled = xtrain / 255.0
xtest_scaled = xtest / 255.0
print(ytrain)
# Model definition

data_augmentation = keras.models.Sequential([
    keras.layers.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    keras.layers.RandomContrast(0.1),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1)
])

model = keras.models.Sequential([
    data_augmentation,
    keras.layers.Conv2D(filters=60, kernel_size=(3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(filters=35, kernel_size=(3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation="relu"),
    keras.layers.Dense(num_class, activation="softmax")
])
'''

base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable =True  # Freeze the base model
fine_tune_at = 100  # Unfreeze from a specific layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = keras.models.Sequential([
    data_augmentation,
    base_model,
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_class, activation="softmax")
])
'''
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(xtrain_scaled, ytrain,epochs=50, validation_split=0.2, callbacks=[early_stopping])
print(ytest)
ypred=model.predict(xtest_scaled)
ypp=[np.argmax(ele) for ele in ypred]
print(classification_report(ytest,ypp))
print(confusion_matrix(ytest,ypp))
'''
# Visualize intermediate outputs
_ = model.predict(xtrain_scaled[:1])
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(xtrain_scaled[:1])
first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()
'''
# Save the model
try:
    model.save("c:/facerecog/facerecognition/registerform/ml/recognition_cnn.keras")
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")
'''=================================check======================================='''
def preprocess(img_path_test):
    img_test=cv2.imread(str(img_path_test))
    gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    face_test=face_cascade.detectMultiScale(gray_test,1.1,4)
    if len(face_test)>0:
        (x_coord,y_coord,w,h)=face_test[0]
        face_idnt=img_test[y_coord:y_coord+h,x_coord:x_coord+w]
    resize=cv2.resize(face_idnt,(224,224))
    resize=cv2.cvtColor(resize,cv2.COLOR_BGR2RGB)
    resized=resize/255
    #plt.imshow(resized)
    #plt.show()
    resized=np.expand_dims(resized,axis=0)
    return resized
pred1=preprocess(file_paths[0])
pred2=preprocess(file_paths[7])

conpred=model.predict(pred1)
conpred2=model.predict(pred2)
cpp1=[np.argmax(ele) for ele in conpred]
cpp2=[np.argmax(ele) for ele in conpred2]
print(cpp1)
print(cpp2)

checkattendance=pathlib.Path("C:/facerecog/facerecognition/temp/user_attendance/user_attendance.jpg")
resized1=preprocess(checkattendance)
Prediction=model.predict(resized1)
pred=[np.argmax(ele) for ele in Prediction]
print(Prediction)
print(pred)