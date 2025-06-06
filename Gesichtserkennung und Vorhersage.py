import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Gesichtsdetektionsmodell laden ---
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# --- Achtung: Diese Modelle und Encoder müssen vorher geladen sein ---
# facenet_model: Modell, das Embeddings aus Gesichtern erzeugt
# model: trainiertes Klassifikationsmodell (z.B. SVM, RandomForest)
# out_encoder: Label-Encoder, um Klassennummern in Namen umzuwandeln
# datasetWahl: int, der angibt, welche Datenbasis verwendet wird

def get_Prediction(filename, bildIndexImDiagramm):
    path = "ValidationBilder/"
    subpath = path + filename

    # Bild laden und in RGB konvertieren
    image = Image.open(subpath)
    image = image.convert('RGB')
    image = np.asarray(image)
    height, width = image.shape[:2]

    # Blob für Gesichtsdetektion vorbereiten
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    # Prüfen, ob mindestens ein Gesicht erkannt wurde
    if faces.shape[2] == 0:
        print(f"Kein Gesicht in {filename} gefunden.")
        return

    # Erstes Gesicht ausschneiden
    box = faces[0, 0, 0, 3:7] * np.array([width, height, width, height])
    (x, y, x1, y1) = box.astype("int")

    # Ränder prüfen, damit Indexe nicht negativ oder außerhalb sind
    x, y = max(0, x), max(0, y)
    x1, y1 = min(width, x1), min(height, y1)

    face_crop = image[y:y1, x:x1]

    # Gesicht auf 160x160 bringen (FaceNet Eingabegröße)
    face = cv2.resize(face_crop, (160, 160))

    # Batch-Dimension hinzufügen, da FaceNet das erwartet
    sample = np.expand_dims(face, axis=0)

    # Embeddings mit facenet_model erzeugen
    samples_2d = facenet_model.embeddings(sample)
    samples = np.expand_dims(samples_2d, axis=0)[0]
    samples_n = np.reshape(samples, (1, 512))  # Annahme: 512 Features

    # Klassenvorhersage und Wahrscheinlichkeiten berechnen
    yhat_class = model.predict(samples_n)
    yhat_prob = model.predict_proba(samples_n)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    predict_name = out_encoder.inverse_transform(yhat_class)

    # Gesicht und Vorhersage in Plot darstellen
    fig.add_subplot(rows, columns, bildIndexImDiagramm)
    plt.imshow(face_crop)
    title = (f'Predictions based on {datasetName} Datatest\n'
             f'Filename: {filename}\n'
             f'Predicted: {predict_name[0]}\n'
             f'Probability: {class_probability:.2f}%')
    plt.title(title)
    plt.axis('off')

# --- Hauptprogramm ---

# Liste der zu testenden Bilder (2 Personen)
filenames = [
    "Saad_Belasri_ohne_Maske.jpg",
    "Saad_Belasri_mit_Maske.jpg"
]

# Dataset-Name bestimmen
if datasetWahl == 1:
    datasetName = 'Unmasked'
elif datasetWahl == 2:
    datasetName = 'Masked'
elif datasetWahl == 3:
    datasetName = 'Mixed - 100'
elif datasetWahl == 4:
    datasetName = 'Mixed - 300'
else:
    datasetName = 'Unknown Dataset'

# Plot mit 1 Zeile, 2 Spalten (für 2 Bilder)
columns = 2
rows = 1
fig = plt.figure(figsize=(12, 6))

# Vorhersagen für alle Bilder erzeugen und plotten
for i, filename in enumerate(filenames, start=1):
    get_Prediction(filename, i)

plt.tight_layout()
plt.savefig("Vorhersagen_Plot.png", dpi=300, bbox_inches='tight')
plt.show()
