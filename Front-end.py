import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from signs import classes
import numpy
# load the trained model to classify sign
from keras.models import load_model
model = load_model('my_model (1).h5')

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#bad5ff')

label = Label(top, background='#bad5ff', font=('arial', 25, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32, 32))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#364156', text=("Result: "+sign))


def show_classify_button(file_path):
    classify_b = Button(top, text="  Classify Image  ",
                        command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156',
                         foreground='#364156', font=('arial', 20, 'bold'))
    classify_b.place(relx=0.362, rely=0.925)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(
            ((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an image",
                command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='#364156',
                 font=('arial', 20, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Traffic Sign Recognition",
                pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#bad5ff', foreground='#364156')

heading.pack()
top.mainloop()
