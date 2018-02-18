__author__ = 'Sandra'

from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import Image, ImageTk
import detector
import tkMessageBox
from keras.models import load_model
import numpy
import webcam
import videoProcesing


emotions = ["Happy", "Sad", "Angry", "Neutral", "Fear", "Surprise", "Disgust"]
emotions_ws = ["Happy", "Angry", "Neutral"]


FILE_PATH = ""

DATA = []

colors = []

EMOTIONS_PATH = "emotions/"

LABELS = []

START_BUTTONS = []

OTHER_BUTTONS = []

model_name = "model23.h5"


class Window(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent, background="#66c2ff")
        self.pack(expand=YES, fill=BOTH)
        canv = Canvas(self, bg="#99d6ff", relief=SUNKEN)
        canv.config(width=1000, height=500)
        canv.config(scrollregion=(0, 0, 0, 2000))
        canv.config(highlightthickness=0)

        sbar = Scrollbar(self)
        sbar.config(command=canv.yview)

        canv.config(yscrollcommand=sbar.set)
        sbar.pack(side=RIGHT, fill=Y)
        canv.pack(side=LEFT, expand=YES, fill=BOTH)


        self.parent = parent
        self.canvas = canv


        self.initUI()



    def browse(self):

        if len(LABELS) != 0:
            for label in LABELS:
                label.destroy()

        filename = askopenfilename()
        global FILE_PATH
        FILE_PATH = filename

        try:
            img = Image.open(filename)
            width, height = img.size
            if width >= 300 or height >= 300:
                new_width = 300
                new_height = float(height)/(float(width)/300)
            else:
                new_height = height
                new_width = width
            img = img.resize((int(new_width), int(new_height)), Image.ANTIALIAS)
            img1 = ImageTk.PhotoImage(img)
            label = Label(self.canvas, image=img1)
            LABELS.append(label)
            label.image = img1
            self.canvas.create_window(200, 180, window=label)

            analyze_button = Button(self.canvas, text="Analyze", command=self.analyze)
            LABELS.append(analyze_button)
            self.canvas.create_window(40, 380, window=analyze_button)

        except IOError:
            tkMessageBox.showerror("Error!", "The file you chose isn't an image! Please try again!")

    def browse_video(self):

        if len(LABELS) != 0:
            for label in LABELS:
                label.destroy()

        filename = askopenfilename()

        global FILE_PATH
        FILE_PATH = filename

        predicted = videoProcesing.process_video(FILE_PATH, model_name)

        self.show_predicted(predicted)

    def show_predicted(self, predicted):
        text = ""
        size = len(predicted)
        print size
        for e in emotions:
            print predicted.count(e)
            percent = float(predicted.count(e)) * 100 / float(size)
            print percent
            text += e + " ---> Found: " + str(round(percent, 2)) + "%\n"

            txt_label = Label(self.canvas, text=text, borderwidth=2, relief="ridge")
            LABELS.append(txt_label)
            txt_label.config(font=("Arial", 12))

            self.canvas.create_window(300, 250, window=txt_label)

    def analyze(self):
        print "file path {}".format(FILE_PATH)
        global DATA
        if len(DATA) != 0:
            del DATA[:]

        filename, DATA = detector.show_picture(FILE_PATH)

        if filename is None and DATA is None:
            tkMessageBox.showerror("Error!", "No faces detected! Please try another picture.")

        img = Image.open(filename)
        width, height = img.size
        if width >= 300 or height >= 300:
            new_width = 300
            new_height = float(height) / (float(width) / 300)
        else:
            new_height = height
            new_width = width
        img = img.resize((int(new_width), int(new_height)), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(img)
        label = Label(self.canvas, image=img1)
        LABELS.append(label)
        label.image = img1
        self.canvas.create_window(200, 180, window=label)

        self.show_results(DATA)

    def show_results(self, data):
        model = load_model(model_name)
        list = []
        y = 80

        for d in data:
            print "face"
            coords = d[0:len(d)-1]
            emotion = d[len(d)-1]

            list.append(coords)

        list1 = numpy.asarray(list)

        predictions = model.predict(list1)
        print predictions

        text = ""

        for pp in predictions:
            for n, p in enumerate(pp):

                text += emotions[n] + ": " + str(round(float(p * 100), 2)) + "%\n"

            predicted_emotion = numpy.argmax(pp)

            text2 = ""
            text2 += "Predicted emotion: " + emotions[predicted_emotion]
            img = Image.open(EMOTIONS_PATH + emotions[predicted_emotion].lower() + ".png")

            txt_label = Label(self.canvas, text=text, borderwidth=2, relief="ridge")
            LABELS.append(txt_label)
            txt_label.config(font=("Arial", 12))

            self.canvas.create_window(700, 20+y, window=txt_label)

            txt_label2 = Label(self.canvas, text=text2, borderwidth=2, relief="ridge")
            LABELS.append(txt_label2)
            txt_label2.config(font=("Arial", 14))

            self.canvas.create_window(700, 120+y, window=txt_label2)

            img = img.resize((100, 100), Image.ANTIALIAS)
            img1 = ImageTk.PhotoImage(img)
            label = Label(self.canvas, image=img1)
            LABELS.append(label)
            label.image = img1

            self.canvas.create_window(900, 80+y, window=label)

            text = ""
            text2 = ""
            y+= 200

    def picture(self):

        for button in START_BUTTONS:
            button.destroy()

        browse_button = Button(self.canvas, text="Browse", command=self.browse)
        self.canvas.create_window(120, 20, window=browse_button)

        back_button = Button(self.canvas, text="Go back", command=self.initUI)
        self.canvas.create_window(40, 20, window=back_button)

        OTHER_BUTTONS.append(back_button)
        OTHER_BUTTONS.append(browse_button)

    def video(self):

        for button in START_BUTTONS:
            button.destroy()

        browse_video_button = Button(self.canvas, text="Browse", command=self.browse_video)
        self.canvas.create_window(120, 20, window=browse_video_button)

        back_button = Button(self.canvas, text="Go back", command=self.initUI)
        self.canvas.create_window(40, 20, window=back_button)

        OTHER_BUTTONS.append(back_button)
        OTHER_BUTTONS.append(browse_video_button)


    def live(self):
        webcam.go_live(model_name)

    def initUI(self):
        self.parent.title("Emotion Detector")

        if len(LABELS) != 0:
            for label in LABELS:
                label.destroy()

        if len(OTHER_BUTTONS) != 0:
            for bb in OTHER_BUTTONS:
                bb.destroy()

        picture_button = Button(self.canvas, text="Picture", command=self.picture, width=15, height=5)
        picture_button.place(x = 150, y = 230)
        self.canvas.create_window(250, 230, window=picture_button)

        video_button = Button(self.canvas, text="Video", command=self.video, width=15, height=5)
        video_button.place(x=400, y=230)
        self.canvas.create_window(500, 230, window=video_button)

        live_button = Button(self.canvas, text="Live", command=self.live, width=15, height=5)
        live_button.place(x=650, y=230)
        self.canvas.create_window(750, 230, window=live_button)

        START_BUTTONS.append(picture_button)
        START_BUTTONS.append(video_button)
        START_BUTTONS.append(live_button)

def main():
    root = Tk()
    root.geometry("1000x500+10+10")

    app = Window(root)
    app.pack()
    root.mainloop()


if __name__ == '__main__':
    main()
