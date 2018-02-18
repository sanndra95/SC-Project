
# Import required modules
import cv2
import dlib
import numpy
from skimage import io
import time
import csv
import os
import random

TRAINING_PATH = "D:/Sandra/Faks/4/Soft/soft_datasets/selenium_dataset/Training"
TESTING_PATH = "D:/Sandra/Faks/4/Soft/soft_datasets/selenium_dataset/Testing"
EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise", "disgust"]

MODE = "TESTING"

dataset_matrix = []

no_faces = []
gray_pics = []

def load_dataset_into_csv(folder_path):
    list_of_paths = os.listdir(folder_path)
    print len(list_of_paths)
    for path in list_of_paths:
        full_path = folder_path + "/" + path
        print full_path
        detect(full_path)

    print len(dataset_matrix)

    write_to_csv(dataset_matrix)

def get_random_color():
    return random.randint(0, 255)

def detect(path):

    if len(dataset_matrix) != 0:
        del dataset_matrix[:]

    try:
        frame = io.imread(path)
        detector = dlib.get_frontal_face_detector()  # Face detector
        predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")

        if len(frame.shape) == 2:
            gray_pics.append(path)
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray)

        detections = detector(clahe_image, 1)  # Detect the faces in the image

        print "Number of faces: {}".format(len(detections))

        if len(detections) == 0:
            no_faces.append(path)
            return None, None

        #lists for x and y coordinates
        xlist = [[] for i in range (len(detections))]
        ylist = [[] for i in range (len(detections))]

        #lists for mean of x and y coordinates --> gets center of the face
        xmean = [[] for i in range (len(detections))]
        ymean = [[] for i in range (len(detections))]

        #matrixes of xlist and ylist for each face
        matrix = [[] for i in range (len(detections))]

        #distances of each landmark from the center of the face
        xcenter = [[] for i in range (len(detections))]
        ycenter = [[] for i in range (len(detections))]

        #relative coordinates in range [0,1]
        xnorm = [[] for i in range(len(detections))]
        ynorm = [[] for i in range(len(detections))]

        #lists for relative coordinates of means
        xmean_norm = [[] for i in range(len(detections))]
        ymean_norm = [[] for i in range(len(detections))]

        # lists for relative distances from the center of the face
        xcenter_norm = [[] for i in range(len(detections))]
        ycenter_norm = [[] for i in range(len(detections))]

        #matrixes of final results (contains lists: xcenter, ycenter, eucl_distances, angles)
        final = [[] for i in range (len(detections))]


        for k, d in enumerate(detections):  # For each detected face

            shape = predictor(clahe_image, d)
            for i in range(0, 68):  # 68 landmarks
                xlist[k].append(shape.part(i).x)
                ylist[k].append(shape.part(i).y)
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (255, 0, 0),
                           thickness=2)  # draw red dots


            xmean[k] = numpy.mean(xlist[k]) #mean of x coordinates
            ymean[k] = numpy.mean(ylist[k]) #mean of y coordinates

            matrix[k] = numpy.column_stack((xlist[k], ylist[k]))
            for x, y in matrix[k]:
                cv2.line(frame, (x, y), (int(xmean[k]), int(ymean[k])), (0, 255, 0), thickness=1) #draw lines

            cv2.circle(frame, (int(xmean[k]), int(ymean[k])), 1, (0, 0, 255),
                       thickness=2)  # draw center dot

            xcenter[k] = ([x - xmean[k] for x in xlist[k]]) #distances from the
            ycenter[k] = ([y - ymean[k] for y in ylist[k]]) #center of the face

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

            emotion = write_emotion(path)
            final[k].append(emotion)

            dataset_matrix.append(final[k])

        return frame, dataset_matrix

    except:
        print "Could not load image!"

def write_emotion(path):
    path = path.split("/")
    pp = path[len(path)-1].split(".")[0]
    print pp
    if "happy" in pp:
        return 0
    elif "sad" in pp:
        return 1
    elif "angry" in pp:
        return 2
    elif "neutral" in pp:
        return 3
    elif "fear" in pp:
        return 4
    elif "surprise" in pp:
        return 5
    elif "disgust" in pp:
        return 6
    else:
        return -1


def show_picture(path):
    splitted = path.split(".")
    mills = int(round(time.time() * 1000))
    new_path = splitted[0] + "_" + str(mills) + "." + splitted[1]
    frame, data = detect(path)
    if frame is None and data is None:
        return None, None
    io.imsave(new_path, frame)
    return new_path, data

def write_to_csv(dataset):
    path = "datasets/"

    if MODE == "TRAINING":
        csv_name = path + "dataset_selenium.csv"
    elif MODE == "TESTING":
        csv_name = path + "testdata_selenium.csv"
    with open(csv_name, 'ab+') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for data in dataset:
            writer.writerow(data)

def main():
    if MODE == "TRAINING":
        start_time = time.time()
        load_dataset_into_csv(TRAINING_PATH)
        end_time = time.time()
    elif MODE == "TESTING":
        start_time = time.time()
        load_dataset_into_csv(TESTING_PATH)
        end_time = time.time()
    print "Vreme za upisivanje u csv fajl: {}".format(end_time - start_time)

    print ""
    print "No faces: " + str(len(no_faces))
    for img in no_faces:
        print img

    print ""
    print "Gray images: " + str(len(gray_pics))
    for pic in gray_pics:
        print pic


if __name__ == "__main__":
    main()
