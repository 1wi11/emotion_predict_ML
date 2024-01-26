from tkinter import *
import os


root = Tk()

root.title("main")
root.attributes("-fullscreen", True)
root.resizable(False, False)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

def run_other_code():
    os.system("cam.py")

l = Label(root, text=" ", font=("휴먼매직체", 60))
l.pack(side="top")

photo_sdj = PhotoImage(file="stopdoljin1_removebg.png")
label_sdj = Label(root, image=photo_sdj)
label_sdj.place(x=screen_width*(1/5), y=50)

w = Label(root, text="WITH", font=("휴먼매직체", 80))
w.place(x=screen_width/2+50, y=100)

photo_gs = PhotoImage(file="gs25.png").subsample(3)
label_gs = Label(root, image=photo_gs)
label_gs.place(x=screen_width/2+50, y=200)



label_title1 = Label(root, text="안녕하세요!", font=("휴먼매직체", 60), fg="blue")
label_title2 = Label(root, text="당신의 기분에 따라 음식을 추천해드립니다!", font=("휴먼매직체", 60), fg="blue")

label_title1.place(x=screen_width*0.4, y=screen_height*0.4)
label_title2.place(x=screen_width*0.15, y=screen_height/2)

def on_button_click():
    run_other_code()

def out():
    root.destroy()


photo_cart = PhotoImage(file="cart.png").subsample(2)
btn_cart = Button(root, image=photo_cart, command=on_button_click)
btn_cart.place(x=screen_width*0.4, y=screen_height*0.65)

btn_out = Button(root, text="나가기", font=("휴먼매직체", 30), command=out)
btn_out.pack(side="bottom")



root.mainloop()