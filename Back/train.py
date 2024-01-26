import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
NUM_CLASSES = 5

dataset = './model/keypoint_classifier/keypoint_classifier_label.csv'
model_save_path = './model/keypoint_classifier/keypoint_classifier.hdf5'
tflite_save_path = './model/keypoint_classifier/keypoint_classifier.tflite'

# Load data from CSV, skipping the first row
data = np.loadtxt(dataset, delimiter=',', dtype='float32', skiprows=1)

# Separate input (X) and output (y) data
X_dataset = data[:, 1:]
y_dataset = data[:, 0].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.8, random_state=RANDOM_SEED)

def shuffle_dataset(X, y):
    unique_labels = np.unique(y)
    num_labels = len(unique_labels)
    num_samples_per_label = len(y) // num_labels

    shuffled_indices = []
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        np.random.shuffle(label_indices)
        shuffled_indices.extend(label_indices[:num_samples_per_label])

    np.random.shuffle(shuffled_indices)
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    return X_shuffled, y_shuffled

# Shuffle the dataset
X_train, y_train = shuffle_dataset(X_train, y_train)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((468 * 2, )),
    tf.keras.layers.Dense(20, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='elu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Model checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
# Callback for early stopping
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# Model compilation
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)

# Model evaluation
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
print("Overall Accuracy:", val_acc)

# Loading the saved model
model = tf.keras.models.load_model(model_save_path)

# Inference test
predicted_classes = np.argmax(model.predict(X_test), axis=1)
for emotion in range(NUM_CLASSES):
    emotion_indices = np.where(y_test == emotion)[0]
    emotion_accuracy = np.sum(predicted_classes[emotion_indices] == y_test[emotion_indices]) / len(emotion_indices)
    print(f"Accuracy for {emotion} emotion: {emotion_accuracy}")

train_predictions = np.argmax(model.predict(X_train), axis=1)
for emotion in range(NUM_CLASSES):
    emotion_indices = np.where(y_train == emotion)[0]
    emotion_accuracy = np.sum(train_predictions[emotion_indices] == y_train[emotion_indices]) / len(emotion_indices)
    print(f"Training Accuracy for {emotion} emotion: {emotion_accuracy}")
