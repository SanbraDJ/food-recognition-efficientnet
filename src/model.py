"""
Módulo para crear y configurar el modelo EfficientNet - OPTIMIZADO
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

def create_efficientnet_model(num_classes=101, img_size=224, trainable_layers=20):
    """
    Crea un modelo basado en EfficientNetV2 con regularización mejorada
    
    Args:
        num_classes: Número de clases de alimentos
        img_size: Tamaño de entrada de la imagen
        trainable_layers: Número de capas a entrenar (desde el final)
    
    Returns:
        model, base_model
    """
    # Cargar modelo base pre-entrenado
    base_model = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    
    # Congelar todas las capas inicialmente
    base_model.trainable = False
    
    # Construcción del modelo con REGULARIZACIÓN MEJORADA
    inputs = keras.Input(shape=(img_size, img_size, 3))
    
    # Base model
    x = base_model(inputs, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout más agresivo
    x = layers.Dropout(0.5)(x)
    
    # Dense layer con L2 regularization
    x = layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=l2(0.01),
        name='dense_512'
    )(x)
    
    # Batch Normalization para estabilidad
    x = layers.BatchNormalization()(x)
    
    # Segundo dropout
    x = layers.Dropout(0.4)(x)
    
    # Dense layer adicional
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=l2(0.01),
        name='dense_256'
    )(x)
    
    # Batch Normalization
    x = layers.BatchNormalization()(x)
    
    # Dropout final
    x = layers.Dropout(0.3)(x)
    
    # Capa de salida
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=l2(0.005),
        name='predictions'
    )(x)
    
    # Crear modelo
    model = keras.Model(inputs, outputs, name='EfficientNetV2_FoodRecognition')
    
    return model, base_model


def unfreeze_model(base_model, trainable_layers=20):
    """
    Descongela las últimas capas del modelo base para fine-tuning
    
    Args:
        base_model: Modelo base de EfficientNet
        trainable_layers: Número de capas a descongelar desde el final
    """
    base_model.trainable = True
    
    # Congelar todas menos las últimas N capas
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    print(f"✅ Descongeladas las últimas {trainable_layers} capas del modelo base")
    print(f"📊 Capas entrenables: {sum([1 for layer in base_model.layers if layer.trainable])}")
    print(f"📊 Capas congeladas: {sum([1 for layer in base_model.layers if not layer.trainable])}")


def load_trained_model(model_path):
    """
    Carga un modelo entrenado desde disco
    
    Args:
        model_path: Ruta al archivo del modelo (.h5 o .keras)
    
    Returns:
        Modelo cargado
    """
    model = keras.models.load_model(model_path)
    print(f"✅ Modelo cargado desde: {model_path}")
    return model


def get_model_summary(model):
    """Muestra un resumen del modelo"""
    print("\n" + "="*70)
    print("📋 RESUMEN DEL MODELO")
    print("="*70)
    model.summary()
    
    # Contar parámetros
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    print(f"\n📊 Parámetros entrenables: {trainable_params:,}")
    print(f"📊 Parámetros no entrenables: {non_trainable_params:,}")
    print(f"📊 Total de parámetros: {trainable_params + non_trainable_params:,}")
    print("="*70 + "\n")