import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Training Image Processing
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    interpolation = "bilinear",
)

# Validation Image Processing
validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    interpolation = "bilinear"
)

# Display a batch shape for debugging
for x, y in training_set:
    print(x.shape)
    print(y.shape)
    break

## Building Model
model = tf.keras.Sequential()

# Convolution Layers
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128, 128, 3]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())  # Fixed typo
model.add(tf.keras.layers.Dense(units=1500, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))

# Output Layer
model.add(tf.keras.layers.Dense(units=38, activation='softmax'))

## Compiling Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

#Traning Model
training_history = model.fit(x=training_set,validation_data=validation_set,epochs = 10)

#Model Evaluation on Traning set
train_loss,train_acc = model.evaluate(training_set)
print(train_loss,train_acc)

#Model on Validation set
val_loss,val_acc = model.evaluate(validation_set)
print(val_loss,val_acc)

##Saving Model
model.save("trained_model.keras")
training_history.history

#Recording History
import json
with open("training_hist.json","w") as f:
    json.dump(training_history.history,f)

#Accuracy Visualization
epochs = [i for i in range (1,11)]
plt.plot(epochs,training_history.history['accuracy'], color = 'red', label = 'Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'], color = 'blue', label = 'Validation Accuracy')
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy Result")
plt.title("Visualization of Accuracy Result")
plt.legend()
plt.show()

#Some other metrics for model
class_name = validation_set.class_names 

test_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed = None,
    validation_split = None,
    subset = None,
    interpolation = "bilinear",
    follow_links = False
    )

y_pred = model.predict(test_set)
y_pred,y_pred.shape

predicted_categories = tf.argmax(y_pred, axis=1)
predicted_categories

true_categories = tf.concat([y for x,y in test_set], axis = 0)
true_categories

y_true = tf.argmax(true_categories, axis=1)
y_true

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, predicted_categories,target_name = class_name))

cm = confusion_matrix(y_true,predicted_categories)
cm.shape

#Confusion Matrix Visualization
plt.figure(figsize=(40,40))
sns = sns.heatmap(cm,annot = True, annot_kws = {'size': 10})
plt.xlabel("Predicted Class", fontsize = 20)
plt.ylabel("Actual Class", fontsize = 20)
plt.title("Plant Decision Prediction Confusion Matrix", fontsize = 25)
plt.show