#importing all the libraries
from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox as mb
import tkinter.font as font
from tkinter import filedialog as fd
import ultralytics
from ultralytics import YOLO
import numpy
import shutil
import os
j = 0
fileToCopy = ''

#main window creation
root = Tk()
root.title("Computer Vision Project")
width= root.winfo_screenwidth()
height= root.winfo_screenheight()
root.geometry("%dx%d" % (width, height))


#font style
myFont = font.Font(family = 'Helvetica',size = 30, weight= "bold")
myFont1 = font.Font(family = 'Helvetica',size = 20, weight= "bold")

#Background image
img = Image.open('../WallpaperDog.jpg')
bg = ImageTk.PhotoImage(img)
lbl = Label(root, image = bg, bg='pink')
lbl.config(bg = "black", fg = "white")
lbl.place(x=0, y=0)

#Text on main window
lbl1 = Label(root, text = "          Y O L O V 8          ", font= myFont)
lbl1.config(bg = "orange")
lbl1.place(x=550,y=10)


#function for Upload button
def buttonfunction():
    global fileToCopy
    fileToCopy = fd.askopenfilename(initialdir="/", title = "Select a File", filetypes = (("Jpg files", "*.jpg.*"), ("Jpeg files", "*.jpeg.*"), ("all files", "*.*")))
    destination= 'D:/Python IDE/ComputerVisionInterface/inputdata' #Enter your destination path


    try:
        shutil.copy(fileToCopy, destination)
        mb.showinfo(
                title="File Uploaded!",
                message="The selected file has been uploaded."
            )


    except:
        mb.showerror(
            title="Error!",
            message="Selected file is unable to upload. Please try again!"
        )


 #Upload button
b = Button(root, text = "Upload File", bg='salmon', font= myFont1, command = buttonfunction)
b.place(x=200, y= 200)


def ibutton(i):
    global a
    a = i

#function for Download button
def buttonfunction1():
    global source_folder
    i = a
    if(i == 1):
        source_folder = 'D:/Python IDE/ComputerVisionInterface/runs/detect/predict'
    if(i == 2):
        source_folder = 'D:/Python IDE/ComputerVisionInterface/runs/classify/predict'
    if (i == 3):
        source_folder = 'D:/Python IDE/ComputerVisionInterface/runs/pose/predict'
    if (i == 4):
        source_folder = 'D:/Python IDE/ComputerVisionInterface/runs/segment/predict'

    destination_folder = 'C:/Users/Gowrilatha/Downloads' #enter your download directory
    destination_folder1 = 'C:/Users/Gowrilatha/Downloads/predict' #enter your download directory
    global j
    j += 1
    if (os.path.exists(destination_folder1)):
        destination_folder1 = destination_folder1 + f"{j}"
        shutil.move(source_folder, destination_folder1)
        mb.showinfo(title="File Downloaded!",
                    message=f"The selected file has been Downloaded. Check out {destination_folder1} folder"
                    )
    else:
        shutil.move(source_folder, destination_folder)
        mb.showinfo(title="File Downloaded!",
                    message=f"The selected file has been Downloaded. Check out {destination_folder} folder"
                    )





#function for Convert button
def buttonfunction2():
    #window for convert tab
    root1 = Toplevel(root)
    root1.title("Convert")
    width = root1.winfo_screenwidth()
    height = root1.winfo_screenheight()
    root1.geometry("%dx%d" % (width, height))
    myFont = font.Font(family='Helvetica', size=30, weight="bold")
    myFont1 = font.Font(family='Helvetica', size=20, weight="bold")
    myFont2 = font.Font(family='Helvetica', size=15, weight="bold")

    img = Image.open('../WallpaperDog.jpg')
    bg = ImageTk.PhotoImage(img)
    lbl = Label(root1, image=bg, bg='pink')
    lbl.config(bg="black", fg="white")
    lbl.place(x=0, y=0)

    lbl1 = Label(root1, text="          Y O L O V 8          ", font=myFont)
    lbl1.config(bg="orange")
    lbl1.place(x=850, y=250)

#this function enables Download button
    def main():
        root1.destroy()


    def buttonfunction4():
        i=1
        if (fileToCopy == ''):
            mb.showinfo(
                title="File not Selected!",
                message="The File has not been uploaded. Please upload the file"
            )
        model = YOLO("../yolov8n.pt", "v8")
        det = model.predict( conf = 0.25, save = True, source = fileToCopy)
        b = Button(root1, text="Go To Download Page", bg='salmon', font=myFont2, command=main)
        b.place(x=950, y=310)
        ibutton(i)

    def buttonfunction5():
        i=2
        if (fileToCopy == ''):
            mb.showinfo(
                title="File not Selected!",
                message="The File has not been uploaded. Please upload the file"
            )
        model = YOLO("../yolov8n-cls.pt", "v8")
        det = model(source = fileToCopy , conf = 0.25, save = True)
        b = Button(root1, text="Go To Download Page", bg='salmon', font=myFont2, command=main)
        b.place(x=950, y=310)
        ibutton(i)

    def buttonfunction6():
        i=3
        if (fileToCopy == ''):
            mb.showinfo(
                title="File not Selected!",
                message="The File has not been uploaded. Please upload the file"
            )
        model = YOLO("../yolov8n-pose.pt", "v8")
        det = model(source = fileToCopy, conf = 0.25, save = True)
        b = Button(root1, text="Go To Download Page", bg='salmon', font=myFont2, command=main)
        b.place(x=950, y=310)
        ibutton(i)

    def buttonfunction7():
        i=4
        if (fileToCopy == ''):
            mb.showinfo(
                title="File not Selected!",
                message="The File has not been uploaded. Please upload the file"
            )
        model = YOLO("../yolov8n-seg.pt", "v8")
        det = model(source = fileToCopy, conf = 0.25, save = True)
        b = Button(root1, text="Go To Download Page", bg='salmon', font=myFont2, command=main)
        b.place(x=950, y=310)
        ibutton(i)

    b = Button(root1, text="Detect Objects", bg='salmon', font=myFont1, command=buttonfunction4)
    b.place(x=200, y=100)

    b = Button(root1, text="Classify Objects", bg='salmon', font=myFont1, command=buttonfunction5)
    b.place(x=200, y=200)

    b = Button(root1, text="Pose of Objects", bg='salmon', font=myFont1, command=buttonfunction6)
    b.place(x=200, y=300)

    b = Button(root1, text="Segment Objects", bg='salmon', font=myFont1, command=buttonfunction7)
    b.place(x=200, y=400)

    root1.mainloop()


b = Button(root, text = "Download File",bg='salmon', font= myFont1,  command = buttonfunction1)
b.place(x=1100, y= 200)

b = Button(root, text = "Convert File", bg='salmon', font= myFont1, command = buttonfunction2)
b.place(x=650, y= 300)

root.mainloop()

