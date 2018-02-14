__author__ = 'Sandra'

from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import Image, ImageTk
import detector
import tkMessageBox
import neural_network
import numpy


emotions = ["Happy", "Sad", "Angry", "Neutral"]
emotions_ws = ["Happy", "Angry", "Neutral"]


FILE_PATH = ""

EMOTIONS_PATH = "emotions/"

LABELS = []


class Window(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent, background="pink")

        self.parent = parent


        self.initUI()

    def browse(self):

        if len(LABELS) != 0:
            for label in LABELS:
                label.destroy()

        #Tk().withdraw()
        filename = askopenfilename()
        print filename
        global FILE_PATH
        FILE_PATH = filename

        try:
            img = Image.open(filename)
            width, height = img.size
            print height, width
            if height >= 300:
                new_height = 300
                new_width = float(width)/(float(height)/300)
                print new_height, new_width
            else:
                new_height = height
                new_width = width
            img = img.resize((int(new_width), new_height), Image.ANTIALIAS)
            img1 = ImageTk.PhotoImage(img)
            label = Label(self, image=img1)
            LABELS.append(label)
            label.image = img1
            label.place(x=20, y=60)
            analyze_button = Button(self, text="Analyze", command=self.analyze)
            LABELS.append(analyze_button)
            analyze_button.place(x=20, y=380)
        except IOError:
            tkMessageBox.showerror("Error!", "The file you chose isn't an image! Please try again!")

    def analyze(self):
        print "file path {}".format(FILE_PATH)

        filename, data = detector.show_picture(FILE_PATH)

        if filename is None and data is None:
            tkMessageBox.showerror("Error!", "No faces detected! Please try another picture.")

        #print "data: {}".format(data)

        img = Image.open(filename)
        width, height = img.size
        print height, width
        if height >= 300:
            new_height = 300
            new_width = float(width) / (float(height) / 300)
            print new_height, new_width
        else:
            new_height = height
            new_width = width
        img = img.resize((int(new_width), new_height), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(img)
        label = Label(self, image=img1)
        LABELS.append(label)
        label.image = img1
        label.place(x=500, y=50)

        self.show_results(data)



    def show_results(self, data):
        model = neural_network.start_function()
        list = []
        if len(data) == 1:
            dd = data[0]
            coords = dd[0:len(dd)-1]
            emotion = dd[len(dd)-1]
        else:
            tkMessageBox.showerror("Error!", "More than one face was detected on the picture! Please try another one.")
            return

        print "coords: {}".format(coords)

        list.append(coords)
        list1 = numpy.asarray(list)
        print "list: {}".format(list1)

        predictions = model.predict(list1)
        print predictions

        text = ""

        pp = predictions[0]
        for n, p in enumerate(pp):
            if neural_network.SAD_MODE:
                text += emotions_ws[n] + ": " + str(round(float(p * 100), 2)) + "%\n"
            else:
                text += emotions[n] + ": " + str(round(float(p * 100), 2)) + "%\n"

        predicted_emotion = numpy.argmax(predictions)

        text2 = ""
        if neural_network.SAD_MODE:
            text2 += "Predicted emotion: " + emotions_ws[predicted_emotion]
            img = Image.open(EMOTIONS_PATH + emotions_ws[predicted_emotion].lower() + ".png")
        else:
            text2 += "Predicted emotion: " + emotions[predicted_emotion]
            img = Image.open(EMOTIONS_PATH + emotions[predicted_emotion].lower() + ".png")

        txt_label = Label(self, text=text, borderwidth=2, relief="ridge")
        LABELS.append(txt_label)
        txt_label.config(font=("Arial", 12))
        txt_label.place(x=500, y=400)

        txt_label2 = Label(self, text=text2, borderwidth=2, relief="ridge")
        LABELS.append(txt_label2)
        txt_label2.config(font=("Arial", 14))
        txt_label2.place(x=500, y=480)

        img = img.resize((100, 100), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(img)
        label = Label(self, image=img1)
        LABELS.append(label)
        label.image = img1
        label.place(x=780, y=400)


    def initUI(self):
        self.parent.title("Emotion Detector")
        self.pack(fill=BOTH, expand=1)

        browse_button = Button(self, text="Browse", command=self.browse)
        browse_button.place(x = 20, y = 20)

def main():
    root = Tk()
    root.geometry("1000x600+100+100")
    app = Window(root)
    root.mainloop()


if __name__ == '__main__':
    main()