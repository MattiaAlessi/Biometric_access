import cv2
import os
from trainer import train_model
from capture_user import capture_images

# Percorsi
dataset_path = "dataset"
model_path = "recognizer.yml"

# 🔧 Funzione per decidere se riaddestrare
def needs_training(model_path, dataset_path):
    if not os.path.exists(model_path):
        return True
    model_time = os.path.getmtime(model_path)
    for user in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user)
        for img in os.listdir(user_path):
            img_path = os.path.join(user_path, img)
            if os.path.getmtime(img_path) > model_time:
                return True
    return False

# 🔧 Crea la cartella dataset se non esiste
os.makedirs(dataset_path, exist_ok=True)

# 👤 Onboarding utente
user_name = input("👤 Inserisci il tuo nome per la registrazione biometrica: ").strip()
user_folder = os.path.join(dataset_path, user_name)

# 📸 Cattura immagini se non esistono
existing_images = os.listdir(user_folder) if os.path.exists(user_folder) else []
if len(existing_images) >= 5:
    print(f"📁 Trovate {len(existing_images)} immagini per {user_name}. Salto la cattura.")
else:
    print(f"📸 Avvio cattura per {user_name}...")
    capture_images(user_name, output_dir=dataset_path, num_images=10)

# 🧠 Addestra il modello LBPH solo se necessario
if needs_training(model_path, dataset_path):
    print("⚙️ Dataset aggiornato. Riaddestro il modello...")
    train_model(dataset_path, model_path)
else:
    print("📁 Modello aggiornato. Salto l'addestramento.")

# 🔍 Riconoscimento in tempo reale
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

label_map = {0: user_name}  # semplificato per utente singolo
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

print("🔍 Posizionati per il riconoscimento...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi)
        name = label_map.get(label, "Unknown")

        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        if confidence < 80:
            print(f"✅ Accesso consentito: {name} (confidenza: {confidence:.2f})")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        else:
            print(f"❌ Accesso negato (confidenza: {confidence:.2f})")

    cv2.imshow("FaceGate", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
