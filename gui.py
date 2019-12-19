from tkinter import *
import sqlite3
from tkinter import ttk
from time import sleep
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
from flowchart_generator import *
from txt_to_word import *
from split_it_image import *
from image_doc import *
from PIL import Image
import pytesseract
root = Tk()
root.geometry('500x500')
root.title("Hand Drawn")
import shutil
'''
Fullname=StringVar()
Email=StringVar()
var1 = IntVar()
c=StringVar()
'''
var= IntVar()
var.set(1)

teams = range(100)
def alert_popup(title,message):
    """Generate a pop-up window for special messages."""
    root = Tk()
    root.title(title)
    w = 400     # popup window width
    h = 200     # popup window height
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w)/2
    y = (sh - h)/2
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    m = message
    m += '\n'
   # m += path
    w = Label(root, text=m, width=50, height=10)
    w.pack()
    b = Button(root, text="OK", command=root.destroy, width=5)
    b.pack()
def database():
   pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe' 
   str12=''
   try: 
       str12=root.filename
   except:
       alert_popup('Error message!!!','Select a file first')
       return    
   if str12=='':
        alert_popup('Error message!!!','Select a file first')
        return
       
   
   files = [  
             ('Docs Files', '*.docx'), 
             ('Text Document', '*.txt'),
             ('All Files', '*.*')]
   gender=var.get()
   if gender!=1:
       files = [  
             ('Docs Files', '*.docx'), 
             ('Image file', '*.jpg'),
             ('All Files', '*.*')]
       
   file = filedialog.asksaveasfile(filetypes = files, defaultextension = files)     
  # print(file.name)
   print(2)
   name1=file.name
  
       
   popup = tk.Toplevel()
   tk.Label(popup, text="Processing..").grid(row=0,column=0)

   progress = 0
   progress_var = tk.DoubleVar()
   progress_bar = ttk.Progressbar(popup, variable=progress_var, maximum=100)
   progress_bar.grid(row=1, column=0)#.pack(fill=tk.X, expand=1, side=tk.BOTTOM)
   popup.pack_slaves()
   i=0
   progress_step = float(100.0/len(teams))
   for team in teams:
       i=i+1
       popup.update()
       sleep(.05) # lauch task
       progress += progress_step
       progress_var.set(progress)
       if i==28:
            if gender==1:
                #print('dddd')
                #print(name1)
                
                call1(str12,name1)
                
                im = Image.open(str12)
                text = pytesseract.image_to_string(im, lang = 'eng')
                f1= open('output/output4.txt',"a")                                      
                f1.write(text)
                f1.close()   
                
                #print(text)
                if name1[len(name1)-1] == 'x':
                   # print('ok')
                    dell(name1,1)
                else:
                    dell(name1,2)
            else:
                temp=call2(str12)
                
   popup.destroy()  
   if gender!=1 :  
       if name1[len(name1)-1] == 'x':
           dell12(temp,name1)
       else:
           shutil.copy("output.jpg",name1)
   root.filename=''
   alert_popup('Completed','Your task has been completed..')
   remove11()
   os.startfile(name1) 
      
#def NewFile():
    #print ("New File!")
def OpenFile():
    root.filename =  filedialog.askopenfilename(initialdir = "I:/handdrawn/draw/Demo/images",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))
    str12=root.filename
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
    im = cv2.imread(str12)                        # Read image
    imS = cv2.resize(im, (2060, 740))                    # Resize image
    cv2.imshow("output", imS)                            # Show image
    cv2.waitKey(0)
def About():
     popup = tk.Toplevel()
     tk.Label(popup, text="Handwriting recognition (HWR), \n also known as Handwritten Text Recognition (HTR), \nis the ability of a computer to receive and interpret \nintelligible handwritten input from sources \nsuch as paper documents, photographs,\n touch-screens and other devices.").grid(row=10,column=1)
def show():
     root.filename =  filedialog.askopenfilename(initialdir = "I:/handdrawn/draw/Demo/images",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))
     str12=root.filename
def remove11():
    try: 
        os.remove("output.jpg")
        os.remove("sou.jpg")
        os.remove("0kalu.jpg")
        os.remove("0para.jpg")
        os.remove("ocr_gray.jpg")
        os.remove("cropped.jpg")
    except: pass  
if __name__== "__main__" :   
    remove11()
    menu = Menu(root)
    root.config(menu=menu)
    filemenu = Menu(menu)
    menu.add_cascade(label="File", menu=filemenu)
    #filemenu.add_command(label="New", command=NewFile)
    filemenu.add_command(label="Open...", command=OpenFile)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=root.destroy)
    
    helpmenu = Menu(menu)
    menu.add_cascade(label="Help", menu=helpmenu)
    helpmenu.add_command(label="About...", command=About)
                
    label_0 = Label(root, text="Hand Drawn",width=20,font=("bold", 20))
    label_0.place(x=90,y=53)
    
    
    #label_1 = Label(root, text="Image name",width=20,font=("bold", 10))
    #label_1.place(x=200,y=130)
    
    #entry_1 = Entry(root,textvar=Fullname)
    #entry_1.place(x=215,y=155)
    
    
    #label_2 = Label(root, text="Choose one",width=20,font=("bold", 10))
    #label_2.place(x=245,y=250)
    
    Radiobutton(root, text="Handwriting recognition",padx = 5, variable=var, value=1,indicatoron = 0).place(x=200,y=260)
    Radiobutton(root, text="Flow chart",padx = 20, variable=var, value=2,indicatoron = 0).place(x=215,y=300)
    
    #label_3 = Label(root, text="File Format",width=20,font=("bold", 10))
    #label_3.place(x=70,y=310)
    
    #list1 = ['JPG','PNG'];
    
    #droplist=OptionMenu(root,c, *list1)
    #droplist.config(width=15)
    #c.set('File Format') 
    #droplist.place(x=210,y=150)
    #droplist.place(x=210,y=190)
    
    Button(root, text='Select an image',width=20,bg='blue',fg='white',command=show).place(x=195,y=155)
    
    Button(root, text='Generate',width=20,bg='brown',fg='white',command=database).place(x=195,y=380)
    # we don't want a full GUI, so keep the root window from appearing
    
    
    
    root.mainloop()
