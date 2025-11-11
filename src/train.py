import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, TensorBoard

# Model parameters
epochs_phase1 = 15
epochs_phase2 = 35
batch_size = 16
learning_rate_phase1 = 0.0001
learning_rate_phase2 = 0.00001

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max'),
    TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
]

# Create the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate_phase1), loss='sparse_categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

# Train the model - Phase 1
model.fit(x_train, y_train, epochs=epochs_phase1, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)

# Train the model - Phase 2 with reduced learning rate
model.optimizer.lr.assign(learning_rate_phase2)
model.fit(x_train, y_train, epochs=epochs_phase2, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)

# Validation messages
val_accuracy = model.evaluate(x_val, y_val)[1]
if val_accuracy >= 0.97:
    print(f'Validation accuracy is {val_accuracy:.2f}, target reached!')
else:
    print(f'Validation accuracy is {val_accuracy:.2f}, target not reached.')