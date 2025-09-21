import cv2
import time
import os

def capture_images(user_name, output_dir="dataset", num_images=1000, interval=0.01):
    user_path = os.path.join(output_dir, user_name)
    os.makedirs(user_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"📸 Posizionati davanti alla webcam. Verranno scattate {num_images} foto...")
    print("Durante gli scatti avvicinati e allontanati dalla telecamera, inoltre prova a girare il volto affinche il modello LBPH sia più preciso")

    count = 0
    last_capture = time.time()

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Face Capture", frame)
        

        # Scatta ogni intervallo secondi
        if time.time() - last_capture >= interval:
            img_path = os.path.join(user_path, f"img{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"✅ Foto {count+1} salvata")
            count += 1
            last_capture = time.time()


    cap.release()
    cv2.destroyAllWindows()
    print("🎉 Acquisizione completata.")
