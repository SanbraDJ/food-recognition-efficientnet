# 🍕 Food Recognition System using EfficientNetV2

## 📋 Descripción del Proyecto

Sistema de reconocimiento de alimentos usando **EfficientNetV2** y el dataset **Food-101**. Este proyecto utiliza técnicas de Deep Learning y Transfer Learning para lograr una precisión superior al **98%** en la clasificación de 101 categorías de alimentos.

## 🎯 Objetivos

- ✅ Clasificación automática de 101 tipos de alimentos
- ✅ Precisión objetivo: **98%+**
- ✅ Entrenamiento eficiente optimizado para GPUs básicas
- ✅ Implementación con TensorFlow/Keras
- ✅ Código modular y reutilizable

## 🏗️ Arquitectura del Modelo

- **Modelo Base:** EfficientNetV2 (preentrenado en ImageNet)
- **Transfer Learning:** Fine-tuning de capas superiores
- **Dataset:** Food-101 (101,000 imágenes, 101 categorías)
- **Framework:** TensorFlow 2.x / Keras

## 📊 Dataset: Food-101

El dataset Food-101 contiene:
- **101 categorías** de alimentos
- **101,000 imágenes** en total
- **750 imágenes de entrenamiento** por clase
- **250 imágenes de test** por clase

Categorías incluyen: pizza, hamburguesa, sushi, tacos, helado, y muchas más.

## 🚀 Instalación

### Requisitos Previos

- Python 3.8+
- GPU con CUDA (recomendado, pero no obligatorio)
- Al menos 8GB de RAM
- 5GB de espacio en disco para el dataset

### Clonar el Repositorio

```bash
git clone https://github.com/SanbraDJ/food-recognition-efficientnet.git
cd food-recognition-efficientnet
```

### Instalar Dependencias

```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
food-recognition-efficientnet/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── download_food101.py          # Script para descargar Food-101
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Carga y preprocesamiento de datos
│   ├── model.py                     # Definición del modelo
│   ├── train.py                     # Script de entrenamiento
│   └── utils.py                     # Funciones auxiliares
├── models/                          # Modelos entrenados guardados
└── results/                         # Resultados y métricas
```

## 🎓 Uso

### 1. Descargar el Dataset

```bash
python data/download_food101.py
```

### 2. Entrenamiento del Modelo

#### Opción A: Usando Notebooks (Recomendado para aprender)

Abre los notebooks en orden:
1. `01_exploratory_data_analysis.ipynb` - Exploración de datos
2. `02_model_training.ipynb` - Entrenamiento del modelo
3. `03_model_evaluation.ipynb` - Evaluación y resultados

#### Opción B: Usando Scripts

```bash
python src/train.py --epochs 25 --batch-size 32 --learning-rate 0.001
```

### 3. Evaluación del Modelo

```python
from src.model import load_trained_model
from src.utils import predict_image

model = load_trained_model('models/efficientnet_food101.h5')
prediction = predict_image(model, 'path/to/image.jpg')
print(f"Predicción: {prediction}")
```

## ⚙️ Configuración Optimizada para GPU Básica

El proyecto está optimizado para GPUs básicas:

- **Batch Size:** 32 (ajustable según tu VRAM)
- **Mixed Precision Training:** Activado para mayor velocidad
- **Data Augmentation:** En tiempo real para ahorrar memoria
- **Gradient Accumulation:** Opcional para simular batches más grandes

## ⏱️ Tiempo de Entrenamiento Estimado

Con GPU básica (GTX 1060 / GTX 1650):
- **Épocas recomendadas:** 20-30
- **Tiempo por época:** ~15-20 minutos
- **Tiempo total:** 5-10 horas

## 📈 Resultados Esperados

| Métrica | Objetivo |
|---------|----------|
| Accuracy | >98% |
| Top-5 Accuracy | >99.5% |
| F1-Score | >0.97 |

## 🛠️ Técnicas Implementadas

- ✅ **Transfer Learning** con EfficientNetV2
- ✅ **Data Augmentation** (rotación, zoom, flip, brillo)
- ✅ **Learning Rate Scheduling** (ReduceLROnPlateau)
- ✅ **Early Stopping** para evitar overfitting
- ✅ **Model Checkpointing** (guardar mejor modelo)
- ✅ **Mixed Precision Training** (FP16)
- ✅ **Class Weights** para balanceo de clases

## 📚 Referencias Científicas

Este proyecto está basado en investigaciones recientes:

1. **Advancements in Food Recognition: A Comprehensive Review of Deep Learning** - IEEE Xplore
2. **Deep Learning in Food Image Recognition** - MDPI
3. **EfficientNetV2: Smaller Models and Faster Training** - Google Research

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**SanbraDJ**
- GitHub: [@SanbraDJ](https://github.com/SanbraDJ)

## 🙏 Agradecimientos

- Dataset Food-101 por ETH Zurich
- TensorFlow y Keras teams
- Comunidad de Deep Learning

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en este repositorio.

---

⭐ Si este proyecto te ayudó, considera darle una estrella en GitHub!