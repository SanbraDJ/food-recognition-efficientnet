"""
Script principal de entrenamiento OPTIMIZADO para 99% precisión
"""
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import argparse
import sys

# Importar módulos locales
from data_loader import Food101DataLoader
from model import create_efficientnet_model, unfreeze_model, get_model_summary
from utils import plot_training_history, save_results, check_gpu, enable_mixed_precision

def train_model(data_dir='data/food-101', 
                epochs_phase1=15, 
                epochs_phase2=35,
                batch_size=16,
                learning_rate_phase1=0.0001,
                learning_rate_phase2=0.00001,
                use_mixed_precision=False):
    """
    Entrena el modelo en dos fases OPTIMIZADO:
    Fase 1: Transfer Learning (solo capas superiores)
    Fase 2: Fine-tuning (todo el modelo)
    
    Args:
        data_dir: Directorio del dataset Food-101
        epochs_phase1: Épocas para fase 1 (incrementado a 15)
        epochs_phase2: Épocas para fase 2 (incrementado a 35)
        batch_size: Tamaño del batch (reducido a 16 para mejor generalización)
        learning_rate_phase1: Learning rate para fase 1 (reducido)
        learning_rate_phase2: Learning rate para fase 2 (reducido)
        use_mixed_precision: Usar precisión mixta FP16
    """
    
    print("\n" + "="*70)
    print("🍕 ENTRENAMIENTO OPTIMIZADO - RECONOCIMIENTO DE ALIMENTOS 🍔")
    print("="*70 + "\n")
    
    # Verificar GPU
    has_gpu = check_gpu()
    
    # Activar precisión mixta si se solicita y hay GPU
    if use_mixed_precision and has_gpu:
        enable_mixed_precision()
        print("⚡ Precisión mixta FP16 ACTIVADA\n")
    
    # Crear directorios necesarios
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # ==================== CARGAR DATOS ====================
    print("\n📊 Cargando datos con augmentation mejorado...")
    data_loader = Food101DataLoader(
        data_dir=data_dir,
        img_size=224,
        batch_size=batch_size
    )
    
    train_ds, val_ds, test_ds = data_loader.create_datasets(validation_split=0.2)
    print("✅ Datos cargados exitosamente\n")
    
    # ==================== CREAR MODELO ====================
    print("🏗️  Creando modelo EfficientNetV2 con regularización mejorada...")
    model, base_model = create_efficientnet_model(
        num_classes=data_loader.num_classes,
        img_size=224,
        trainable_layers=30  # Más capas para mejor fine-tuning
    )
    
    get_model_summary(model)
    
    # ==================== FASE 1: TRANSFER LEARNING ====================
    print("\n" + "="*70)
    print("🎯 FASE 1: TRANSFER LEARNING (OPTIMIZADO)")
    print("="*70)
    print("Entrenando solo las capas superiores (clasificador)...\n")
    
    # Compilar modelo para fase 1 con learning rate optimizado
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate_phase1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )
    
    # Callbacks MEJORADOS para fase 1
    callbacks_phase1 = [
        keras.callbacks.ModelCheckpoint(
            'models/best_model_phase1.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,  # Más paciencia
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # Reducción más agresiva
            patience=3,
            min_lr=1e-8,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='logs/phase1',
            histogram_freq=1
        )
    ]
    
    # Entrenar fase 1
    print("🚀 Iniciando entrenamiento Fase 1...")
    print(f"⚙️  Configuración:")
    print(f"   - Épocas: {epochs_phase1}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Learning rate: {learning_rate_phase1}")
    print()
    
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Evaluación fase 1
    val_loss_p1, val_acc_p1, val_top5_p1 = model.evaluate(val_ds, verbose=0)
    print(f"\n✅ Fase 1 completada!")
    print(f"   📊 Val Accuracy: {val_acc_p1*100:.2f}%")
    print(f"   📊 Val Top-5 Accuracy: {val_top5_p1*100:.2f}%\n")
    
    # ==================== FASE 2: FINE-TUNING ====================
    print("\n" + "="*70)
    print("🎯 FASE 2: FINE-TUNING (OPTIMIZADO)")
    print("="*70)
    print("Entrenando todo el modelo con learning rate muy bajo...\n")
    
    # Descongelar capas del modelo base (más capas)
    unfreeze_model(base_model, trainable_layers=30)
    
    # Recompilar con learning rate MUY bajo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate_phase2),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )
    
    # Callbacks MEJORADOS para fase 2
    callbacks_phase2 = [
        keras.callbacks.ModelCheckpoint(
            'models/best_model_final.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # Mucha más paciencia
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-9,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='logs/phase2',
            histogram_freq=1
        )
    ]
    
    # Entrenar fase 2
    print("🚀 Iniciando entrenamiento Fase 2...")
    print(f"⚙️  Configuración:")
    print(f"   - Épocas: {epochs_phase2}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Learning rate: {learning_rate_phase2}")
    print()
    
    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase2,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    print("\n✅ Fase 2 completada!\n")
    
    # ==================== EVALUACIÓN FINAL ====================
    print("\n" + "="*70)
    print("📈 EVALUACIÓN FINAL EN TEST SET")
    print("="*70 + "\n")
    
    test_loss, test_accuracy, test_top5 = model.evaluate(test_ds, verbose=1)
    
    print(f"\n🎉 RESULTADOS FINALES:")
    print(f"   🎯 Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   🎯 Test Top-5 Accuracy: {test_top5*100:.2f}%")
    print(f"   📉 Test Loss: {test_loss:.4f}")
    
    # Verificar si alcanzamos el objetivo
    if test_accuracy >= 0.97:
        print("\n   ✅ ¡OBJETIVO ALCANZADO! Precisión >= 97%")
    elif test_accuracy >= 0.95:
        print("\n   ⚠️  Casi ahí... Precisión >= 95%")
    else:
        print("\n   ℹ️  Considera entrenar más épocas o ajustar hiperparámetros")
    
    # ==================== GUARDAR RESULTADOS ====================
    print("\n💾 Guardando resultados...")
    
    # Combinar historias de ambas fases
    combined_history = {
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
        'top5_accuracy': history_phase1.history['top5_accuracy'] + history_phase2.history['top5_accuracy'],
        'val_top5_accuracy': history_phase1.history['val_top5_accuracy'] + history_phase2.history['val_top5_accuracy']
    }
    
    # Crear objeto history combinado
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    combined_hist = CombinedHistory(combined_history)
    
    # Guardar gráficas
    plot_training_history(combined_hist, save_path='results/training_history.png')
    
    # Guardar métricas
    test_metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'test_top5_accuracy': float(test_top5)
    }
    save_results(combined_hist, test_metrics, save_dir='results')
    
    # Guardar modelo final
    model.save('models/food_recognition_final.keras')
    print("✅ Modelo final guardado en: models/food_recognition_final.keras")
    
    print("\n" + "="*70)
    print("🎊 ENTRENAMIENTO COMPLETADO EXITOSAMENTE 🎊")
    print("="*70 + "\n")
    
    return model, combined_hist


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrenar modelo de reconocimiento de alimentos OPTIMIZADO')
    parser.add_argument('--data_dir', type=str, default='data/food-101',
                       help='Directorio del dataset Food-101')
    parser.add_argument('--epochs_phase1', type=int, default=15,
                       help='Épocas para fase 1 (recomendado: 15)')
    parser.add_argument('--epochs_phase2', type=int, default=35,
                       help='Épocas para fase 2 (recomendado: 35)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamaño del batch (recomendado: 16)')
    parser.add_argument('--lr_phase1', type=float, default=0.0001,
                       help='Learning rate para fase 1')
    parser.add_argument('--lr_phase2', type=float, default=0.00001,
                       help='Learning rate para fase 2')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Usar precisión mixta FP16')
    
    args = parser.parse_args()
    
    # Verificar que exista el dataset
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"❌ Error: No se encontró el dataset en {args.data_dir}")
        print("   Por favor ejecuta primero: python data/download_food101.py")
        sys.exit(1)
    
    # Verificar estructura del dataset
    if not (data_path / 'images').exists() or not (data_path / 'meta').exists():
        print(f"❌ Error: Estructura del dataset incompleta en {args.data_dir}")
        print("   Asegúrate de tener las carpetas 'images' y 'meta'")
        sys.exit(1)
    
    # Entrenar modelo
    model, history = train_model(
        data_dir=args.data_dir,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        batch_size=args.batch_size,
        learning_rate_phase1=args.lr_phase1,
        learning_rate_phase2=args.lr_phase2,
        use_mixed_precision=args.mixed_precision
    )
    
    print("✅ Todo listo! Puedes usar el modelo guardado en models/food_recognition_final.keras")


if __name__ == "__main__":
    main()
