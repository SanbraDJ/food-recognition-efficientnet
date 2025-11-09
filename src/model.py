"""
Definición del modelo EfficientNetV2 para clasificación de alimentos
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_efficientnet_model(num_classes=101, img_size=224, trainable_layers=20):
    """
    Crea un modelo EfficientNetV2 con Transfer Learning
    
    Args:
        num_classes: Número de clases (101 para Food-101)
        img_size: Tamaño de entrada de la imagen
        trainable_layers: Número de capas superiores a entrenar
    
    Returns:
        modelo de Keras
    """
    # Cargar modelo base preentrenado
    base_model = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    
    # Congelar capas base inicialmente
    base_model.trainable = False
    
    # Crear el modelo completo
    inputs = keras.Input(shape=(img_size, img_size, 3))
    
    # Augmentation layer (opcional, para mayor robustez)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Pasar por el modelo base
    x = base_model(x, training=False)
    
    # Agregar capas de clasificación
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model

def unfreeze_model(base_model, trainable_layers=20):
    """
    Descongela las últimas capas del modelo base para fine-tuning
    
    Args:
        base_model: Modelo base de EfficientNet
        trainable_layers: Número de capas a descongelar
    """
    base_model.trainable = True
    
    # Congelar todas las capas excepto las últimas
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    print(f"✅ Fine-tuning activado: {trainable_layers} capas entrenables")

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
    """
    Muestra un resumen del modelo
    """
    print("\n" + "="*70)
    print("ARQUITECTURA DEL MODELO")
    print("="*70)
    model.summary()
    print("="*70)
    
    # Contar parámetros entrenables
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    print(f"\n📊 Parámetros entrenables: {trainable_params:,}")
    print(f"📊 Parámetros no entrenables: {non_trainable_params:,}")
    print(f"📊 Total de parámetros: {trainable_params + non_trainable_params:,}\n")
