
from keras.models import load_model
import cv2
import numpy
import dlib

cc = []

emotions = ["Happy", "Sad", "Angry", "Neutral", "Fear", "Surprise", "Disgust"]

all_emotions = []
predicted = []

def detect_video(detections, predictor, clahe_image):
    # lists for x and y coordinates
    xlist = [[] for i in range(len(detections))]
    ylist = [[] for i in range(len(detections))]

    # lists for mean of x and y coordinates --> gets center of the face
    xmean = [[] for i in range(len(detections))]
    ymean = [[] for i in range(len(detections))]

    # matrixes of xlist and ylist for each face
    matrix = [[] for i in range(len(detections))]

    # distances of each landmark from the center of the face
    xcenter = [[] for i in range(len(detections))]
    ycenter = [[] for i in range(len(detections))]

    # relative coordinates in range [0,1]
    xnorm = [[] for i in range(len(detections))]
    ynorm = [[] for i in range(len(detections))]

    # lists for relative coordinates of means
    xmean_norm = [[] for i in range(len(detections))]
    ymean_norm = [[] for i in range(len(detections))]

    # lists for relative distances from the center of the face
    xcenter_norm = [[] for i in range(len(detections))]
    ycenter_norm = [[] for i in range(len(detections))]

    # matrixes of final results (contains lists: xcenter, ycenter, eucl_distances, angles)
    final = [[] for i in range(len(detections))]

    for k, d in enumerate(detections):  # For each detected face

        shape = predictor(clahe_image, d)
        for i in range(0, 68):  # 68 landmarks
            xlist[k].append(shape.part(i).x)
            ylist[k].append(shape.part(i).y)

        xmean[k] = numpy.mean(xlist[k])  # mean of x coordinates
        ymean[k] = numpy.mean(ylist[k])  # mean of y coordinates

        matrix[k] = numpy.column_stack((xlist[k], ylist[k]))

        xcenter[k] = ([x - xmean[k] for x in xlist[k]])  # distances from the
        ycenter[k] = ([y - ymean[k] for y in ylist[k]])  # center of the face

        for i in xlist[k]:
            xnorm[k].append(float((i - min(xlist[k]))) / float((max(xlist[k]) - min(xlist[k]))))

        for i in ylist[k]:
            ynorm[k].append(float((i - min(ylist[k]))) / float((max(ylist[k]) - min(ylist[k]))))

        xmean_norm[k] = float((xmean[k] - min(xlist[k]))) / float((max(xlist[k]) - min(xlist[k])))
        ymean_norm[k] = float((ymean[k] - min(ylist[k]))) / float((max(ylist[k]) - min(ylist[k])))

        #print "Relative distances"
        xcenter_norm[k] = ([x - xmean_norm[k] for x in xnorm[k]])  # relative distances from the
        ycenter_norm[k] = ([y - ymean_norm[k] for y in ynorm[k]])  # center of the face

        for i in range(len(xcenter_norm[k])):
            final[k].append(xcenter_norm[k][i])
            final[k].append(ycenter_norm[k][i])

        if len(cc) != 0:
            del cc[:]

        cc.append(final[k])

    return cc


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0, font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def go_live(model_name):

    model = load_model(model_name)

    video_capture = cv2.VideoCapture(0)  # Webcam object
    detector = dlib.get_frontal_face_detector()  # Face detector
    predictor = dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat")

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray)

        detections = detector(clahe_image, 1)  # Detect the faces in the image

        coords = detect_video(detections, predictor, clahe_image)

        c = numpy.array(coords)

        predictions = model.predict(c)

        text = ""

        pp = predictions[0]
        for n, p in enumerate(pp):
            text += emotions[n] + ": " + str(round(float(p * 100), 2)) + "%\n"

        all_emotions.append(text)

        print ""

        predicted_emotion = numpy.argmax(predictions)

        text2 = ""
        text2 += "Predicted emotion: " + emotions[predicted_emotion]

        print text2

        predicted.append(text2)

        for k, d in enumerate(detections):  # For each detected face

            shape = predictor(clahe_image, d)  # Get coordinates
            for i in range(1, 68):  # There are 68 landmark points on each face
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255),
                           thickness=2)  # For each point, draw a red circle with thickness2 on the original frame

        cv2.imshow("image", frame)  # Display the frame
        font = cv2.QT_FONT_NORMAL
        cv2.putText(frame, text2, (50,50), font, 1, (255, 255, 255), 2)
        cv2.imshow("image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit program when the user presses 'q'
            break


if __name__ == '__main__':
    go_live("model23.h5")