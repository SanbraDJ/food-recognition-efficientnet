"""
Funciones auxiliares para el proyecto de reconocimiento de alimentos
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
def predict_image(model, img_path, classes, top_k=5):
    """
    Predice la clase de una imagen
    
    Args:
        model: Modelo entrenado
        img_path: Ruta a la imagen
        classes: Lista de nombres de clases
        top_k: Número de predicciones top a retornar
    
    Returns:
        Lista de tuplas (clase, probabilidad)
    """
    # Cargar y preprocesar imagen
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    img = tf.expand_dims(img, 0)
    
    # Predecir
    predictions = model.predict(img, verbose=0)
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append((classes[idx], float(predictions[0][idx])))
    
    return results
def plot_training_history(history, save_path=None):
    """
    Grafica la historia de entrenamiento
    
    Args:
        history: Historia de Keras
        save_path: Ruta para guardar la gráfica
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfica guardada en: {save_path}")
    
    plt.show()
def plot_predictions(model, images, labels, classes, num_images=16):
    """
    Muestra predicciones en una grilla de imágenes
    
    Args:
        model: Modelo entrenado
        images: Batch de imágenes
        labels: Etiquetas verdaderas
        classes: Lista de nombres de clases
        num_images: Número de imágenes a mostrar
    """
    predictions = model.predict(images[:num_images], verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    rows = int(np.ceil(num_images / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 4))
    axes = axes.flatten()
    
    for i in range(num_images):
        # Desnormalizar imagen
        img = images[i].numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[i].imshow(img)
        true_label = classes[labels[i]]
        pred_label = classes[pred_classes[i]]
        confidence = predictions[i][pred_classes[i]]
        
        color = 'green' if pred_classes[i] == labels[i] else 'red'
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\n({confidence:.2%})", 
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
def save_results(history, test_metrics, save_dir='results'):
    """
    Guarda los resultados del entrenamiento
    
    Args:
        history: Historia de Keras
        test_metrics: Métricas de evaluación en test
        save_dir: Directorio para guardar resultados
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Guardar historia
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    # Guardar métricas de test
    with open(save_path / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    print(f"✅ Resultados guardados en: {save_path}")
def calculate_class_weights(labels):
    """
    Calcula pesos de clase para datos desbalanceados
    
    Args:
        labels: Array de etiquetas
    
    Returns:
        Diccionario de pesos por clase
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = {int(cls): total / (len(unique) * count) 
               for cls, count in zip(unique, counts)}
    return weights
def enable_mixed_precision():
    """
    Activa el entrenamiento con precisión mixta para mejor rendimiento en GPU
    """
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✅ Precisión mixta activada (FP16)")
    print(f"   Compute dtype: {policy.compute_dtype}")
    print(f"   Variable dtype: {policy.variable_dtype})
def check_gpu():
    """
    Verifica la disponibilidad de GPU
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU disponible: {len(gpus)} dispositivo(s)")
        for gpu in gpus:
            print(f"   - {gpu.name}")
    else:
        print("⚠️  No se detectó GPU. Entrenamiento en CPU (será más lento)")
    
    return len(gpus) > 0