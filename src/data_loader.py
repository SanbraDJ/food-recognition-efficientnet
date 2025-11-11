"""
Módulo para cargar y preprocesar datos de Food-101 - OPTIMIZADO
"""
import tensorflow as tf
from pathlib import Path
import json
import numpy as np

class Food101DataLoader:
    def __init__(self, data_dir='data/food-101', img_size=224, batch_size=32):
        """
        Inicializa el cargador de datos para Food-101
        
        Args:
            data_dir: Directorio donde está el dataset
            img_size: Tamaño de las imágenes (224 para EfficientNet)
            batch_size: Tamaño del batch
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Cargar metadatos
        self.load_metadata()
        
    def load_metadata(self):
        """Carga las clases y splits del dataset"""
        meta_dir = self.data_dir / 'meta'
        
        # Cargar clases
        with open(meta_dir / 'classes.txt', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Cargar splits de entrenamiento y prueba
        with open(meta_dir / 'train.json', 'r') as f:
            self.train_data = json.load(f)
        
        with open(meta_dir / 'test.json', 'r') as f:
            self.test_data = json.load(f)
        
        self.num_classes = len(self.classes)
        print(f"✅ Cargadas {self.num_classes} clases")
        print(f"📊 Imágenes de entrenamiento: {sum(len(v) for v in self.train_data.values())}")
        print(f"📊 Imágenes de prueba: {sum(len(v) for v in self.test_data.values())}")
    
    def create_datasets(self, validation_split=0.2):
        """
        Crea datasets de TensorFlow para entrenamiento, validación y prueba
        
        Args:
            validation_split: Proporción de datos de entrenamiento para validación
        
        Returns:
            train_ds, val_ds, test_ds
        """
        # Crear listas de archivos y etiquetas
        train_files, train_labels = self._create_file_lists(self.train_data)
        test_files, test_labels = self._create_file_lists(self.test_data)
        
        # Shuffle antes de split para mejor distribución
        indices = np.random.permutation(len(train_files))
        train_files = [train_files[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        
        # Split de validación
        num_train = len(train_files)
        num_val = int(num_train * validation_split)
        
        val_files = train_files[:num_val]
        val_labels = train_labels[:num_val]
        train_files = train_files[num_val:]
        train_labels = train_labels[num_val:]
        
        print(f"📊 Split final:")
        print(f"   - Entrenamiento: {len(train_files)} imágenes")
        print(f"   - Validación: {len(val_files)} imágenes")
        print(f"   - Prueba: {len(test_files)} imágenes")
        
        # Crear datasets
        train_ds = self._create_dataset(train_files, train_labels, is_training=True)
        val_ds = self._create_dataset(val_files, val_labels, is_training=False)
        test_ds = self._create_dataset(test_files, test_labels, is_training=False)
        
        return train_ds, val_ds, test_ds
    
    def _create_file_lists(self, data_dict):
        """Crea listas de archivos y etiquetas desde el diccionario de datos"""
        files = []
        labels = []
        
        for class_name, image_list in data_dict.items():
            class_idx = self.classes.index(class_name)
            for img_name in image_list:
                img_path = self.data_dir / 'images' / f"{img_name}.jpg"
                files.append(str(img_path))
                labels.append(class_idx)
        
        return files, labels
    
    def _create_dataset(self, files, labels, is_training=False):
        """Crea un tf.data.Dataset con preprocesamiento"""
        dataset = tf.data.Dataset.from_tensor_slices((files, labels))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.map(
            lambda x, y: self._process_image(x, y, is_training),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _process_image(self, file_path, label, is_training):
        """Carga y preprocesa una imagen con augmentation mejorado"""
        # Leer imagen
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Redimensionar
        img = tf.image.resize(img, [self.img_size, self.img_size])
        
        # Data augmentation MEJORADO para entrenamiento
        if is_training:
            # Flip horizontal
            img = tf.image.random_flip_left_right(img)
            
            # Rotación aleatoria (simulada con crop)
            img = tf.image.random_crop(
                tf.image.resize(img, [int(self.img_size * 1.15), int(self.img_size * 1.15)]),
                [self.img_size, self.img_size, 3]
            )
            
            # Ajustes de color más agresivos
            img = tf.image.random_brightness(img, max_delta=0.3)
            img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
            img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
            img = tf.image.random_hue(img, max_delta=0.15)
            
            # Clip valores para mantener rango válido
            img = tf.clip_by_value(img, 0.0, 255.0)
        
        # Normalizar al rango de EfficientNet
        img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
        
        return img, label