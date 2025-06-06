import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import time

# Absoluter Pfad zu deinem Dataset auf dem Desktop (ändern falls nötig)
TRAIN_DATASET_PATH = r"C:\Users\saadb\Desktop\Dataset_final - Unmasked"
TEST_DATASET_PATH = r"C:\Users\saadb\Desktop\Dataset_final - Test"

# Globale Variable für Anzahl der Personen
anzahl_personen = 0

def lade_bilddaten(basis_pfad):
    """
    Lädt Bilder aus einem Ordner mit Unterordnern für jede Person.
    Jedes Bild wird auf 160x160 skaliert und als RGB-Array zurückgegeben.
    """
    X, Y = [], []
    global anzahl_personen

    if not os.path.exists(basis_pfad):
        raise FileNotFoundError(f"Verzeichnis '{basis_pfad}' existiert nicht.")

    for unterordner in sorted(os.listdir(basis_pfad)):
        ordner_pfad = os.path.join(basis_pfad, unterordner)
        if not os.path.isdir(ordner_pfad):
            continue

        bilder = []
        for datei in sorted(os.listdir(ordner_pfad)):
            bild_pfad = os.path.join(ordner_pfad, datei)
            try:
                bild = Image.open(bild_pfad).convert('RGB')
                bild = bild.resize((160, 160))
                bilder.append(np.array(bild))
            except Exception as e:
                print(f"Warnung: Bild konnte nicht geladen werden: {bild_pfad} -> {e}")

        if len(bilder) > 0:
            X.extend(bilder)
            Y.extend([unterordner] * len(bilder))
            anzahl_personen += 1
            print(f"{len(bilder)} Bilder von Person '{unterordner}' geladen.")
        else:
            print(f"Kein gültiges Bild im Ordner '{unterordner}' gefunden.")

    return np.array(X), np.array(Y)

def main():
    global anzahl_personen

    print("Lade Trainingsbilder...")
    anzahl_personen = 0
    trainX, trainY = lade_bilddaten(TRAIN_DATASET_PATH)
    print(f"Anzahl Personen im Trainingsdatensatz: {anzahl_personen}")
    print(f"trainX shape: {trainX.shape}, trainY shape: {trainY.shape}")

    print("\nLade Testbilder...")
    anzahl_personen = 0
    testX, testY = lade_bilddaten(TEST_DATASET_PATH)
    print(f"Anzahl Personen im Testdatensatz: {anzahl_personen}")
    print(f"testX shape: {testX.shape}, testY shape: {testY.shape}")

    # Lade das vortrainierte FaceNet Modell
    facenet_model = FaceNet()
    print("\nFaceNet Modell geladen.")

    # Berechnung der Embeddings (512-dim Vektoren) für Trainings- und Testbilder
    emdTrainX = facenet_model.embeddings(trainX)
    emdTrainX = np.asarray(emdTrainX)
    print(f"Trainings-Embeddings Shape: {emdTrainX.shape}")

    emdTestX = facenet_model.embeddings(testX)
    emdTestX = np.asarray(emdTestX)
    print(f"Test-Embeddings Shape: {emdTestX.shape}")

    # Normalisierung der Embeddings auf L2-Norm
    normalizer = Normalizer(norm='l2')
    emdTrainX_norm = normalizer.transform(emdTrainX)
    emdTestX_norm = normalizer.transform(emdTestX)
    print("Embeddings normalisiert.")

    # Label-Encoding der Personen-Namen (Strings -> Zahlen)
    label_encoder = LabelEncoder()
    label_encoder.fit(trainY)
    trainY_enc = label_encoder.transform(trainY)
    testY_enc = label_encoder.transform(testY)
    print("Labels codiert.")

    # Visualisierung der Embeddings mit TSNE (2D)
    print("Erstelle TSNE-Plot...")
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)
    z = tsne.fit_transform(emdTrainX_norm)

    df = pd.DataFrame()
    df["label"] = trainY_enc
    df["x"] = z[:, 0]
    df["y"] = z[:, 1]

    plt.figure(figsize=(20, 20))
    sns.scatterplot(x="x", y="y", hue=df.label.tolist(),
                    palette=sns.color_palette("hls", len(label_encoder.classes_)),
                    data=df, legend='full')
    plt.title("TSNE-Visualisierung der FaceNet-Embeddings")
    plt.savefig("facenet_tsne_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Training eines linearen SVM-Klassifikators auf den Embeddings
    print("Trainiere SVM-Klassifikator...")
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(emdTrainX_norm, trainY_enc)

    # Vorhersagen auf Testdaten und Zeitmessung
    print("Führe Vorhersagen auf Testdaten durch...")
    startzeit = time.time()
    y_pred = svm_model.predict(emdTestX_norm)
    endzeit = time.time()

    dauer = endzeit - startzeit
    durchschnitt_zeit = dauer / len(emdTestX_norm)

    print(f"Gesamtdauer der Vorhersage: {dauer:.4f} Sekunden")
    print(f"Durchschnittliche Zeit pro Vorhersage: {durchschnitt_zeit:.6f} Sekunden")

    # Konfusionsmatrix erzeugen und anzeigen
    cm = confusion_matrix(testY_enc, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

    plt.figure(figsize=(12, 12))
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Konfusionsmatrix der Gesichtserkennung")
    plt.savefig("facenet_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Klassifikationsbericht ausgeben
    print("\nKlassifikationsbericht:")
    print(classification_report(testY_enc, y_pred, target_names=label_encoder.classes_))

if __name__ == "__main__":
    main()
