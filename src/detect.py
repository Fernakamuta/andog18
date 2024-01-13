import cv2
import numpy as np
import simpleaudio as sa
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("andog18_valpha.h5")

FPS_CAPTURED = 1
LAILA_THRESHOLD = 0.01
SOUND_FILE = "fart-08.wav"


def play_sound(sound_file):
    wave_obj = sa.WaveObject.from_wave_file(sound_file)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until sound has finished playing


def preprocess_opencv_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = preprocess_input(frame)

    return np.expand_dims(frame, axis=0)


def main():
    # Initialize video capture on the first camera
    print("🎥 Initializing video capture...")
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cap.set(cv2.CAP_PROP_FPS, FPS_CAPTURED)

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            break

        # Display the resulting frame

        frame_preprocessed = preprocess_opencv_frame(frame)
        prediction = model.predict(frame_preprocessed)

        prediction = prediction[0][0]

        label = (
            f"Laila encontrada p={round(prediction, 2)}"
            if prediction >= 0.8
            else f"Laila nao encontrada p={round(prediction, 2)}"
        )

        if prediction >= LAILA_THRESHOLD:
            print("Playing sound")
            play_sound(SOUND_FILE)

        # Display the label on the frame
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video Feed", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
