from tkinter import *
import sqlite3
from flowchart_generator import *
from txt_to_word import *
from image_doc import *
from image_partition import *
root = Tk()
root.geometry('500x500')
root.title("Hand Drawn")


Fullname=StringVar()
Email=StringVar()
var = IntVar()
c=StringVar()
var1= IntVar()
var.set(1)


def database():
   name1=Fullname.get()+"."+c.get()
   print(name1)
   gender=var.get()
   if gender==1:
       call1(name1)
       dell()
   else:
       temp=call2(name1)
       dell12(temp) 
   
             
label_0 = Label(root, text="Hand Drawn",width=20,font=("bold", 20))
label_0.place(x=90,y=53)


label_1 = Label(root, text="Image name",width=20,font=("bold", 10))
label_1.place(x=200,y=130)

entry_1 = Entry(root,textvar=Fullname)
entry_1.place(x=215,y=155)


#label_2 = Label(root, text="Choose one",width=20,font=("bold", 10))
#label_2.place(x=245,y=250)

Radiobutton(root, text="Handwriting recognition",padx = 5, variable=var, value=1,indicatoron = 0).place(x=215,y=260)
Radiobutton(root, text="Flow chart",padx = 20, variable=var, value=2,indicatoron = 0).place(x=215,y=300)

#label_3 = Label(root, text="File Format",width=20,font=("bold", 10))
#label_3.place(x=70,y=310)

list1 = ['JPG','PNG'];

droplist=OptionMenu(root,c, *list1)
droplist.config(width=15)
c.set('File Format') 
droplist.place(x=210,y=190)

Button(root, text='Generate',width=20,bg='brown',fg='white',command=database).place(x=215,y=380)

root.mainloop()