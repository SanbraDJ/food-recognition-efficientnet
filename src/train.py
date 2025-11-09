"""
Script principal de entrenamiento para el modelo de reconocimiento de alimentos
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
                epochs_phase1=10, 
                epochs_phase2=20,
                batch_size=32,
                learning_rate_phase1=0.001,
                learning_rate_phase2=0.0001,
                use_mixed_precision=False):
    """
    Entrena el modelo en dos fases:
    Fase 1: Transfer Learning (solo capas superiores)
    Fase 2: Fine-tuning (todo el modelo)
    
    Args:
        data_dir: Directorio del dataset Food-101
        epochs_phase1: Épocas para fase 1
        epochs_phase2: Épocas para fase 2
        batch_size: Tamaño del batch
        learning_rate_phase1: Learning rate para fase 1
        learning_rate_phase2: Learning rate para fase 2
        use_mixed_precision: Usar precisión mixta FP16
    """
    
    print("\n" + "="*70)
    print("🍕 ENTRENAMIENTO DE MODE LO DE RECONOCIMIENTO DE ALIMENTOS 🍔")
    print("="*70 + "\n")
    
    # Verificar GPU
    has_gpu = check_gpu()
    
    # Activar precisión mixta si se solicita
    if use_mixed_precision and has_gpu:
        enable_mixed_precision()
    
    # Crear directorios necesarios
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # ==================== CARGAR DATOS ====================
    print("\n📊 Cargando datos...")
    data_loader = Food101DataLoader(
        data_dir=data_dir,
        img_size=224,
        batch_size=batch_size
    )
    
    train_ds, val_ds, test_ds = data_loader.create_datasets(validation_split=0.2)
    print("✅ Datos cargados exitosamente\n")
    
    # ==================== CREAR MODELO ====================
    print("🏗️  Creando modelo EfficientNetV2...")
    model, base_model = create_efficientnet_model(
        num_classes=data_loader.num_classes,
        img_size=224,
        trainable_layers=20
    )
    
    get_model_summary(model)
    
    # ==================== FASE 1: TRANSFER LEARNING ====================
    print("\n" + "="*70)
    print("🎯 FASE 1: TRANSFER LEARNING")
    print("="*70)
    print("Entrenando solo las capas superiores (clasificador)...\n")
    
    # Compilar modelo para fase 1
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate_phase1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks para fase 1
    callbacks_phase1 = [
        keras.callbacks.ModelCheckpoint(
            'models/best_model_phase1.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Entrenar fase 1
    print("🚀 Iniciando entrenamiento Fase 1...")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    print("\n✅ Fase 1 completada!\n")
    
    # ==================== FASE 2: FINE-TUNING ====================
    print("\n" + "="*70)
    print("🎯 FASE 2: FINE-TUNING")
    print("="*70)
    print("Entrenando todo el modelo con learning rate bajo...\n")
    
    # Descongelar capas del modelo base
    unfreeze_model(base_model, trainable_layers=20)
    
    # Recompilar con learning rate más bajo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate_phase2),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks para fase 2
    callbacks_phase2 = [
        keras.callbacks.ModelCheckpoint(
            'models/best_model_final.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    # Entrenar fase 2
    print("🚀 Iniciando entrenamiento Fase 2...")
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
    
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    
    print(f"\n🎉 RESULTADOS FINALES:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # ==================== GUARDAR RESULTADOS ====================
    print("\n💾 Guardando resultados...")
    
    # Combinar historias de ambas fases
    combined_history = {
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
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
        'test_accuracy': float(test_accuracy)
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
    parser = argparse.ArgumentParser(description='Entrenar modelo de reconocimiento de alimentos')
    parser.add_argument('--data_dir', type=str, default='data/food-101',
                       help='Directorio del dataset Food-101')
    parser.add_argument('--epochs_phase1', type=int, default=10,
                       help='Épocas para fase 1 (transfer learning)')
    parser.add_argument('--epochs_phase2', type=int, default=20,
                       help='Épocas para fase 2 (fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamaño del batch')
    parser.add_argument('--lr_phase1', type=float, default=0.001,
                       help='Learning rate para fase 1')
    parser.add_argument('--lr_phase2', type=float, default=0.0001,
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