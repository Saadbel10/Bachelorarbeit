import cv2  # Bibliothek für Bildverarbeitung
import numpy as np  # Für numerische Operationen, z.B. mit Arrays

# Pfade zu den Modellen (bitte an dein System anpassen!)
model_datei = r"C:\Users\saadb\Desktop\models\res10_300x300_ssd_iter_140000.caffemodel"
config_datei = r"C:\Users\saadb\Desktop\models\deploy.prototxt"

# Laden des vortrainierten Caffe-Modells für Gesichtserkennung
net = cv2.dnn.readNetFromCaffe(config_datei, model_datei)

# Zugriff auf die Webcam (Kamera 0, mit DirectShow-Backend)
kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Setze die Auflösung der Webcam auf Full HD (1920x1080)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Schriftart für Text auf dem Bild
schriftart = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Lese einen Frame von der Kamera ein
    ret, frame = kamera.read()
    
    # Prüfe, ob der Frame erfolgreich gelesen wurde
    if not ret:
        print("Fehler beim Lesen des Kamerabildes.")
        break
    
    # Maße des Bildes ermitteln (Höhe, Breite)
    hoehe, breite = frame.shape[:2]
    
    # Erstelle einen "Blob" aus dem Bild (Vorverarbeitung für das Netzwerk)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 117.0, 123.0))
    
    # Übergib den Blob an das Netzwerk
    net.setInput(blob)
    
    # Netzwerk führt Vorwärtsdurchlauf durch und gibt Gesichter zurück
    gesichter = net.forward()
    
    # Durchlaufe alle erkannten Gesichter
    for i in range(gesichter.shape[2]):
        vertrauen = gesichter[0, 0, i, 2]  # Vertrauensscore
        
        # Nur Gesichter mit hoher Sicherheit verarbeiten
        if vertrauen > 0.8:
            # Bounding-Box (Gesichtsrechteck) berechnen (relativ zu Bildgröße)
            box = gesichter[0, 0, i, 3:7] * np.array([breite, hoehe, breite, hoehe])
            (x, y, x2, y2) = box.astype(int)
            
            # Begrenze Koordinaten auf Bildbereich (damit es keine Fehler gibt)
            x = max(0, x)
            y = max(0, y)
            x2 = min(breite - 1, x2)
            y2 = min(hoehe - 1, y2)
            
            # Zeichne ein rotes Rechteck um das Gesicht
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
            
            # Gesichtsausschnitt extrahieren
            gesicht_auschnitt = frame[y:y2, x:x2]
            
            try:
                # Gesicht auf 160x160 Pixel skalieren (Format für Erkennungsnetz)
                gesicht_resized = cv2.resize(gesicht_auschnitt, (160, 160))
                
                # Hier kann das Embedding berechnet und Klassifikation erfolgen
                # Beispiel:
                # sample = np.expand_dims(gesicht_resized, axis=0)
                # embedding = facenet_model.embeddings(sample)
                # prediction = model.predict(embedding)
                # ...
                
                # Als Platzhalter Text (später ersetzen durch echtes Ergebnis)
                cv2.putText(frame, f"Erkanntes Gesicht #{i+1}", (x, y-10),
                            schriftart, 0.8, (0, 255, 0), 2)
            
            except Exception as e:
                # Fehler beim Skalieren oder Klassifizieren abfangen
                print(f"Fehler bei Gesichtsbearbeitung: {e}")
    
    # Zeige das aktuelle Kamerabild mit Rechtecken an
    cv2.imshow("Saad Belasri FaceID", frame)
    
    # Warte auf Tastendruck 'q' zum Beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
kamera.release()
cv2.destroyAllWindows()
