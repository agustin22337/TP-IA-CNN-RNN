# TP Redes Neuronales - CNN y RNN

Proyecto simple con dos modelos Keras:

- CNN: clasificación de hojas del dataset Beans de Hugging Face.
- RNN: predicción de temperatura con el dataset Jena Climate.

## 1. Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 3. Ejecutar CNN

```bash
python train_cnn.py
```

El script descarga automáticamente el dataset Beans y genera:

- `models/beans_cnn.keras`
- `results/cnn_accuracy.png`
- `results/cnn_loss.png`
- `results/cnn_confusion_matrix.png`

## 4. Ejecutar RNN

```bash
python train_rnn.py
```

El script descarga automáticamente Jena Climate y utiliza:

- Temperatura
- Humedad relativa
- Presión

Se usan las últimas 24 mediciones para predecir la temperatura siguiente.
La partición se hace cronológicamente: 70% train, 15% validation y 15% test.

Genera:

- `models/temperature_rnn.keras`
- `results/rnn_loss.png`
- `results/rnn_predictions.png`

## Arquitectura CNN

- Conv2D 32
- MaxPooling2D
- Conv2D 64
- MaxPooling2D
- Conv2D 128
- MaxPooling2D
- GlobalAveragePooling2D
- Dense 64
- Dropout 0.3
- Dense 3 Softmax

## Arquitectura RNN

- SimpleRNN 32
- Dense 16
- Dense 1
