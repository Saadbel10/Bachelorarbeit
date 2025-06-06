import numpy as np
import cv2
import os

# Schriftart für Textanzeigen im Bildfenster
schrift = cv2.FONT_HERSHEY_SIMPLEX

# Basisverzeichnis für Dataset (auf dem Desktop)
base_dir = r"C:\Users\saadb\Desktop\Dataset_final"

# ----- Namenseingabe mit Bestätigung -----
while True:
    print("Bitte gib den Namen der Person ein oder q zum Beenden:")
    label = input()
    if label == "q":
        exit(0)  # Programm beenden, wenn 'q' eingegeben wurde
    print(f"Ist der Name '{label}' korrekt? (y/n)")
    answer = input().lower()
    if answer == "y":
        break  # Eingabe bestätigen und weiter

# ----- Erstelle Ordner für Trainings- und Testbilder, falls noch nicht vorhanden -----
train_dir = os.path.join(base_dir, "Train", label)
test_dir = os.path.join(base_dir, "Test", label)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ----- Lade vortrainiertes Gesichtsdetektionsmodell (Caffe-Format) -----
modelFile = r"C:\Users\saadb\Desktop\models\res10_300x300_ssd_iter_140000.caffemodel"
configFile = r"C:\Users\saadb\Desktop\models\deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# ----- Webcam initialisieren und Auflösung einstellen -----
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ----- Parameter für Bildaufnahme -----
count = 0
offset = 10  # Anzahl Bilder am Anfang überspringen (z.B. wegen Licht)
nmData_total = 310 + offset  # Gesamtzahl der Bilder (inkl. Offset)
nmData_train = int(nmData_total * 0.8) + offset  # 80% Trainingsbilder (inkl. Offset)
count_train = 0
count_test = 0

# ----- Erste Phase: Gesichter ohne Maske erfassen -----
while count <= nmData_total:
    ret, img = cap.read()
    if not ret:
        print("Fehler: Kamera konnte kein Bild liefern.")
        break

    height, width = img.shape[:2]

    # Bild vorbereiten für das DNN-Modell (Resize, Mean-Subtraktion)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()  # Gesichter erkennen

    i = 0
    confidence = faces[0, 0, i, 2]

    # Nur Gesicht mit hoher Sicherheit verarbeiten
    if confidence > 0.8:
        box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
        (x, y, x1, y1) = box.astype("int")

        # Gesicht ausschneiden aus dem Originalbild
        face = img[y:y1, x:x1]

        # Gesicht mit rotem Rechteck markieren
        cv2.rectangle(img, (x-2, y-2), (x1+2, y1+2), (0, 0, 255), 2)

        # Offset-Bilder überspringen
        if count > offset:
            # Trainingsbilder speichern
            if count <= nmData_train:
                # Beim Start der Trainingsaufnahme wartet das Programm auf 'c'-Taste
                if count == offset + 1:
                    print("Drücke 'c', um mit dem Training zu beginnen.")
                    while True:
                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            break
                count_train += 1
                index = count_train
                try:
                    dateipfad = os.path.join(train_dir, f"{label}_train_{index}.png")
                    cv2.imwrite(dateipfad, face)  # Gesicht speichern
                except Exception as e:
                    print(f"Fehler beim Speichern: {str(e)}")

                # Anzeige der Anzahl gespeicherter Trainingsbilder im Bildfenster
                cv2.putText(img, f"Trainingsbilder: {count_train}", (50, 50), schrift, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Gesamt: {count - offset}", (50, 100), schrift, 1, (255, 255, 255), 2)

            # Testbilder speichern
            else:
                # Warte auf 'c'-Taste, bevor Testbilder aufgenommen werden
                if count == nmData_train + 1:
                    print("Drücke 'c', um mit den Testbildern zu beginnen.")
                    while True:
                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            break
                count_test += 1
                index = count_test
                try:
                    dateipfad = os.path.join(test_dir, f"{label}_test_{index}.png")
                    cv2.imwrite(dateipfad, face)  # Gesicht speichern
                except Exception as e:
                    print(f"Fehler beim Speichern: {str(e)}")

                # Anzeige der Anzahl gespeicherter Testbilder im Bildfenster
                cv2.putText(img, f"Testbilder: {count_test}", (50, 50), schrift, 1, (0, 0, 255), 2)
                cv2.putText(img, f"Gesamt: {count - offset}", (50, 100), schrift, 1, (255, 255, 255), 2)

        count += 1

    # Zeige Live-Video mit Markierungen
    cv2.imshow("Gesichtserfassung ohne Maske", img)

    # Beenden bei 'q' drücken
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Programm beendet durch Benutzer.")
        break

# Kamera und Fenster schließen
cap.release()
cv2.destroyAllWindows()

# ----- Zweite Phase: Gesichter mit Maske erfassen -----

# Kamera neu starten
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Zähler zurücksetzen
count = 0
count_train = 0
count_test = 0

while count <= nmData_total:
    ret, img = cap.read()
    if not ret:
        print("Fehler: Kamera konnte kein Bild liefern.")
        break

    height, width = img.shape[:2]

    # Bild für Gesichtsdetektion vorbereiten
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    i = 0
    confidence = faces[0, 0, i, 2]

    if confidence > 0.8:
        box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
        (x, y, x1, y1) = box.astype("int")

        face = img[y:y1, x:x1]

        # Gesicht mit rotem Rechteck markieren
        cv2.rectangle(img, (x-2, y-2), (x1+2, y1+2), (0, 0, 255), 2)

        if count > offset:
            # Trainingsbilder mit Maske speichern
            if count <= nmData_train:
                if count == offset + 1:
                    print("Drücke 'c', um mit dem Training (Masken) zu beginnen.")
                    while True:
                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            break
                count_train += 1
                # Index anpassen, damit keine Namenskollision mit erster Phase entsteht
                index = count_train + nmData_train - offset
                try:
                    dateipfad = os.path.join(train_dir, f"{label}_train_{index}.png")
                    cv2.imwrite(dateipfad, face)
                except Exception as e:
                    print(f"Fehler beim Speichern: {str(e)}")

                cv2.putText(img, f"Trainingsbilder: {count_train}", (50, 50), schrift, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Gesamt: {count - offset}", (50, 100), schrift, 1, (255, 255, 255), 2)

            # Testbilder mit Maske speichern
            else:
                if count == nmData_train + 1:
                    print("Drücke 'c', um mit den Testbildern (Masken) zu beginnen.")
                    while True:
                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            break
                count_test += 1
                index = count_test + nmData_train - offset
                try:
                    dateipfad = os.path.join(test_dir, f"{label}_test_{index}.png")
                    cv2.imwrite(dateipfad, face)
                except Exception as e:
                    print(f"Fehler beim Speichern: {str(e)}")

                cv2.putText(img, f"Testbilder: {count_test}", (50, 50), schrift, 1, (0, 0, 255), 2)
                cv2.putText(img, f"Gesamt: {count - offset}", (50, 100), schrift, 1, (255, 255, 255), 2)

        count += 1

    # Live-Videofenster mit Rechteck und Text
    cv2.imshow("Gesichtserfassung mit Maske", img)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Programm beendet durch Benutzer.")
        break

cap.release()
cv2.destroyAllWindows()
