from Tkinter import *

def buttonPushed():
    print "button puthsed"

win= Tk()

b1 = Button(win, text="1")
b2= Button(win,text="2",command=buttonPushed)
b1.pack()
b2.pack()

tBox = Entry(win)
tBox.pack()
win.mainloop()