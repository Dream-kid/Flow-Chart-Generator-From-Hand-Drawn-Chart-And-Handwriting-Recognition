from tkinter import *
import PIL
from PIL import ImageTk

root = Tk()


class PokemonClass(object):

    def __init__(self, master):
        frame = Frame(master)
        frame.pack()

        self.WelcomeLabel = Label(root, text = "Welcome! Pick your Pokemon!", bg = "Black", fg = "White")
        self.WelcomeLabel.pack(fill = X)

        self.CharButton = Button(root, text = "Charmander", bg = "RED", fg = "White", command = CharClick)
        self.CharButton.pack(side = LEFT, fill = X)

        self.SquirtButton = Button(root, text = "Squirtle", bg = "Blue", fg = "White")
        self.SquirtButton.pack(side = LEFT, fill = X)

        self.BulbButton = Button(root, text = "Bulbasaur", bg = "Dark Green", fg = "White")
        self.BulbButton.pack(side = LEFT, fill = X)

    def CharClick():
        print ("You like Charmander!")
        CharPhoto =  ImageTk.PhotoImage(file='para.jpg')
        ChLabel = Label(root, image = CharPhoto)
        ChLabel.pack()


k = PokemonClass(root)
root.mainloop()