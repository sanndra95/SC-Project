
import cv2
import numpy
from keras.models import load_model
import dlib
import webcam


emotions = ["Happy", "Sad", "Angry", "Neutral", "Fear", "Surprise", "Disgust"]

all_emotions = []
predicted = []

final = []

list_of_predicted = []


def process_video(file_name, model_name):

    model = load_model(model_name)

    cap = cv2.VideoCapture(file_name)


    detector = dlib.get_frontal_face_detector()  # Face detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while cap.isOpened():
        ret, frame = cap.read()

        try:
            '''height, width, layers = frame.shape
            new_h = height / 2
            new_w = width / 2
            frame = cv2.resize(frame, (new_w, new_h))'''

            frameId = int(round(cap.get(1)))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(gray)

            detections = detector(clahe_image, 1)  # Detect the faces in the image

            if len(detections) != 0:

                coords = webcam.detect_video(detections, predictor, clahe_image)
                c = numpy.array(coords)
                predictions = model.predict(c)

                text = ""

                pp = predictions[0]
                best = 0
                for n, p in enumerate(pp):
                    percent = round(float(p * 100), 2)
                    if percent > best:
                        best = percent
                    text += emotions[n] + ": " + str(round(float(p * 100), 2)) + "%\n"

                print best

                all_emotions.append(text)

                print ""

                predicted_emotion = numpy.argmax(predictions)

                text2 = ""
                text2 += "Predicted: " + emotions[predicted_emotion] + " " + str(best) + "%"

                print text2

                predicted.append(text2)

                list_of_predicted.append(emotions[predicted_emotion])

                for k, d in enumerate(detections):  # For each detected face

                    shape = predictor(clahe_image, d)  # Get coordinates
                    for i in range(1, 68):  # There are 68 landmark points on each face
                        cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255),
                                   thickness=2)  # For each point, draw a red circle with thickness2 on the original frame

                cv2.imshow("image", frame)  # Display the frame
                font = cv2.QT_FONT_NORMAL
                cv2.putText(frame, text2, (20, 20), font, 1, (255, 255, 255), 2)
                cv2.imshow("image", frame)

        except:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return list_of_predicted



