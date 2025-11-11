import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import argparse
import sys

from Food101DataLoader import Food101DataLoader
from create_efficientnet_model import create_efficientnet_model
from unfreeze_model import unfreeze_model
from get_model_summary import get_model_summary
from plot_training_history import plot_training_history
from save_results import save_results
from check_gpu import check_gpu
from enable_mixed_precision import enable_mixed_precision


def train_model(data_dir='data/food-101', epochs_phase1=15, epochs_phase2=35, batch_size=16, learning_rate_phase1=0.0001, learning_rate_phase2=0.00001, use_mixed_precision=False):
    check_gpu()
    if use_mixed_precision:
        enable_mixed_precision()

    # Load data
    data_loader = Food101DataLoader(data_dir)
    train_data, val_data = data_loader.load_data(batch_size)

    # Create model
    model = create_efficientnet_model()

    # Phase 1 - Transfer Learning
    # Freeze layers and compile model
    model.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate_phase1),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

    # Callbacks
    callbacks_phase1 = [
        keras.callbacks.ModelCheckpoint('phase1_best_model.h5', save_best_only=True),
        keras.callbacks.EarlyStopping(patience=7),
        keras.callbacks.ReduceLROnPlateau(factor=0.3),
        keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    # Train model
    history_phase1 = model.fit(train_data,
                                 validation_data=val_data,
                                 epochs=epochs_phase1,
                                 callbacks=callbacks_phase1)

    # Phase 2 - Fine-Tuning
    unfreeze_model(model)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate_phase2),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

    # Callbacks for Phase 2
    callbacks_phase2 = [
        keras.callbacks.ModelCheckpoint('phase2_best_model.h5', save_best_only=True),
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ReduceLROnPlateau(factor=0.2),
        keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    # Train model Phase 2
    history_phase2 = model.fit(train_data,
                                 validation_data=val_data,
                                 epochs=epochs_phase2,
                                 callbacks=callbacks_phase2)

    # Combined history
    combined_history = {key: history_phase1.history[key] + history_phase2.history[key]
                        for key in history_phase1.history.keys()}
    plot_training_history(combined_history)

    # Save results
    save_results(model, combined_history)

    print('Training completed successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/food-101')
    parser.add_argument('--epochs_phase1', type=int, default=15)
    parser.add_argument('--epochs_phase2', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_phase1', type=float, default=0.0001)
    parser.add_argument('--lr_phase2', type=float, default=0.00001)
    parser.add_argument('--use_mixed_precision', action='store_true')
    args = parser.parse_args(sys.argv[1:])

    train_model(data_dir=args.data_dir,
                epochs_phase1=args.epochs_phase1,
                epochs_phase2=args.epochs_phase2,
                batch_size=args.batch_size,
                learning_rate_phase1=args.lr_phase1,
                learning_rate_phase2=args.lr_phase2,
                use_mixed_precision=args.use_mixed_precision)