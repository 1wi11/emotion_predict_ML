from tkinter import *
from tkinter import filedialog
from os import path


root = Tk()

root.title("angry")
root.attributes('-fullscreen', True)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
sketchbook = Canvas(root, width = screen_width, height = screen_height)
root.resizable(False, False)

photo_gs = PhotoImage(file="gs25.png").subsample(5)
label_gs = Label(root, image=photo_gs)
label_gs.place(x=10, y=20)
l = Label(root, text=" ", font=("휴먼매직체", 50))
l.pack(side="top")
label_menuT = Label(root, text="원하시는 메뉴 카테고리를 골라주세요!", font=("휴먼매직체", 50))
label_menuT.pack()


def open_new_window1():
    new_window = NewWindow1(root)
    root.withdraw()

def open_new_window2():
    new_window = NewWindow2(root)
    root.withdraw()

def open_new_window3():
    new_window = NewWindow3(root)
    root.withdraw()  

def open_new_window4():
    new_window = NewWindow4(root)
    root.withdraw()

    
class NewWindow1(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.meal_angry1()
    
    def meal_angry1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        angry_img = PhotoImage(file="angry/angry_meal/1.png")
        img = Label(self, image=angry_img)
        img.place(x=screen_width/2-365, y=screen_height/2-150)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="전주비빔김밥은 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18),  width=20, height=3,bg="skyblue", command=meal_angry2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()

class NewWindow2(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.drink_angry1()
    
    def drink_angry1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        drink_angry1 = PhotoImage(file="angry/angry_drink/1.png")
        label_drink_happy_img = Label(self, image=drink_angry1)
        label_drink_happy_img.place(x=screen_width/2-95, y=screen_height/2-227)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="Monster는 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        label = Label(self, text="<이번 달 신제품이에요!>", font=("맑은 고딕", 30))
        label.pack(side="top")
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_angry2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()

class NewWindow3(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.dessert_angry1()
    
    def dessert_angry1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        dessert_happy1 = PhotoImage(file="angry/angry_dessert/1.png")
        img = Label(self, image=dessert_happy1)
        img.place(x=screen_width/2-257, y=screen_height/2-244)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="초콜릿 케이크는 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
    
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_angry2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()


class NewWindow4(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.none_angry1()
    
    def none_angry1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        none = PhotoImage(file="angry/angry_none/1.png").subsample(2)
        img = Label(self, image=none)
        img.place(x=screen_width/2-350, y=screen_height/2-350)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="매운분식 도시락은 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")

        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_angry2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()


def location_meal1():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    angry_img = PhotoImage(file="angry/angry_meal/1.png")
    img = Label(newWin, image=angry_img)
    img.place(x=screen_width/2-365, y=screen_height/2-150)
    label_lo = Label(newWin, text='"좌측 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")
    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_meal2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    meal_happy2_img = PhotoImage(file="angry/angry_meal/2.png")
    label_meal_happy2_img = Label(newWin, image=meal_happy2_img)
    label_meal_happy2_img.place(x=screen_width/2-235, y=screen_height/2-220)

    label_lo = Label(newWin, text='"라면 코너 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)

    newWin.mainloop()


def location_meal3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    meal_happy3_img = PhotoImage(file="angry/angry_meal/3.png")
    label_meal_happy3_img = Label(newWin, image=meal_happy3_img)
    label_meal_happy3_img.place(x=screen_width/2-205, y=screen_height/2-190)

    label_lo = Label(newWin, text='"오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")
    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)

    newWin.mainloop()

def location_meal4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    meal_happy4_img = PhotoImage(file="angry/angry_meal/4.png")
    label_meal_happy4_img = Label(newWin, image=meal_happy4_img)
    label_meal_happy4_img.place(x=screen_width/2-305, y=screen_height/2-267)

    label_lo = Label(newWin, text='"좌측 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)

    newWin.mainloop()

def location_meal5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    meal_happy5_img = PhotoImage(file="angry/angry_meal/5.png")
    label_meal_happy5_img = Label(newWin, image=meal_happy5_img)
    label_meal_happy5_img.place(x=screen_width/2-260, y=screen_height/2-262)

    label_lo = Label(newWin, text='"라면 코너 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")
    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)

    newWin.mainloop()

def meal_angry2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    meal_happy2_img = PhotoImage(file="angry/angry_meal/2.png")
    label_meal_happy2_img = Label(newWin, image=meal_happy2_img)
    label_meal_happy2_img.place(x=screen_width/2-235, y=screen_height/2-220)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    label_meal_happy2_q = Label(newWin, text="틈새라면은 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy2_q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_angry3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_angry3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    meal_happy3_img = PhotoImage(file="angry/angry_meal/3.png")
    label_meal_happy3_img = Label(newWin, image=meal_happy3_img)
    label_meal_happy3_img.place(x=screen_width/2-205, y=screen_height/2-190)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    label_meal_happy3_q = Label(newWin, text="매콤한 비빔면 샐러드는 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy3_q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_angry4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_angry4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    meal_happy4_img = PhotoImage(file="angry/angry_meal/4.png")
    label_meal_happy4_img = Label(newWin, image=meal_happy4_img)
    label_meal_happy4_img.place(x=screen_width/2-305, y=screen_height/2-267)
    label_meal_happy4_q = Label(newWin, text="뭘. 몰. 다. 도시락은 어떠신가요?",font=("휴먼매직체", 50))
    label_meal_happy4_q.pack(side="top")
    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_angry5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_angry5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    meal_happy5_img = PhotoImage(file="angry/angry_meal/5.png")
    label_meal_happy5_img = Label(newWin, image=meal_happy5_img)
    label_meal_happy5_img.place(x=screen_width/2-260, y=screen_height/2-262)

    label_meal_happy5_q = Label(newWin, text="불닭볶음면은 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy5_q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal5)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=done)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()



def location_drink1():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    drink_img = PhotoImage(file="angry/angry_drink/1.png")
    label_img = Label(newWin, image=drink_img)
    label_img.place(x=screen_width/2-95, y=screen_height/2-227)

    label_lo = Label(newWin, text='"캔음료 코너 중간 냉장고"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()


def location_drink2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    drink_img = PhotoImage(file="angry/angry_drink/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-160, y=screen_height/2-228)

    label_lo = Label(newWin, text='"음료 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_drink3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    drink_img = PhotoImage(file="angry/angry_drink/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-113, y=screen_height/2-212)

    label_lo = Label(newWin, text='"캔음료 코너 중간 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_drink4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    drink_img = PhotoImage(file="angry/angry_drink/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-128, y=screen_height/2-218)

    label_lo = Label(newWin, text='"캔음료 코너 중간 냉장고"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_drink5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    drink_img = PhotoImage(file="angry/angry_drink/5.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-290, y=screen_height/2-233)

    label_lo = Label(newWin, text='"우측 오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def drink_angry2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="angry/angry_drink/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-160, y=screen_height/2-228)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="씨그램 탄산수는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_angry3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_angry3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="angry/angry_drink/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-113, y=screen_height/2-212)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="레전더리 코카콜라는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")
    label = Label(newWin, text="<1+1 행사상품이에요!>", font=("맑은 고딕", 30))
    label.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_angry4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_angry4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="angry/angry_drink/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-128, y=screen_height/2-218)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="닥터페퍼는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_angry5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_angry5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="angry/angry_drink/5.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-290, y=screen_height/2-233)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="제로 소다는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink5)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=done)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()


def location_dessert1():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    dessert_happy1 = PhotoImage(file="angry/angry_dessert/1.png")
    img = Label(newWin, image=dessert_happy1)
    img.place(x=screen_width/2-257, y=screen_height/2-244)

    label_lo = Label(newWin, text='"좌측 오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_dessert2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    drink_img = PhotoImage(file="angry/angry_dessert/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-380, y=screen_height/2-182)


    label_lo = Label(newWin, text='"과자 코너 왼쪽 편 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_dessert3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    drink_img = PhotoImage(file="angry/angry_dessert/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-110, y=screen_height/2-246)

    label_lo = Label(newWin, text='"좌측 오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_dessert4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    drink_img = PhotoImage(file="angry/angry_dessert/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-201, y=screen_height/2-183)

    label_lo = Label(newWin, text='"젤리 코너 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_dessert5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    angry_img = PhotoImage(file="angry/angry_dessert/5.png")
    img = Label(newWin, image=angry_img)
    img.place(x=screen_width/2-278, y=screen_height/2-347)

    label_lo = Label(newWin, text='"젤리 코너 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def dessert_angry2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="angry/angry_dessert/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-380, y=screen_height/2-182)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="와사비맛 아몬드는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_angry3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_angry3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="angry/angry_dessert/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-110, y=screen_height/2-246)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="흑당 크림롤은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_angry4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_angry4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="angry/angry_dessert/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-201, y=screen_height/2-183)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="아이셔는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_angry5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_angry5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    angry_img = PhotoImage(file="angry/angry_dessert/5.png")
    img = Label(newWin, image=angry_img)
    img.place(x=screen_width/2-278, y=screen_height/2-347)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="폭탄 젤리는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert5)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3,bg="skyblue", command=done)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()


def location_none1():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    none = PhotoImage(file="angry/angry_none/1.png").subsample(2)
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-350, y=screen_height/2-350)
    
    label_lo = Label(newWin, text='"오픈 냉장고 우측 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_none2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    none = PhotoImage(file="angry/angry_none/2.png")
    img = Label(newWin, image=none)
    img.place(x=screen_height/2-210, y=screen_height/2-187)

    label_lo = Label(newWin, text='"라면 코너 가운데 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_none3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    none = PhotoImage(file="angry/angry_none/3.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-250, y=screen_height/2-413)

    label_lo = Label(newWin, text='"좌측 오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_none4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    none = PhotoImage(file="angry/angry_none/4.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-290, y=screen_height/2-165)

    label_lo = Label(newWin, text='"좌측 오픈 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_none5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    angry_img = PhotoImage(file="angry/angry_none/5.png")
    img = Label(newWin, image=angry_img)
    img.place(x=screen_width/2-367, y=screen_height/2-350)

    label_lo = Label(newWin, text='"오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def none_angry2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="angry/angry_none/2.png")
    img = Label(newWin, image=none)
    img.place(x=screen_height/2-210, y=screen_height/2-187)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="챔라면은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")
    label = Label(newWin, text="<이번 달 신제품이에요!>", font=("맑은 고딕", 30))
    label.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_angry3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_angry3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="angry/angry_none/3.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-250, y=screen_height/2-413)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="월드콘 케이크는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")
    label = Label(newWin, text="<현재 2+1 행사상품이에요!>", font=("맑은 고딕", 30))
    label.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_angry4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_angry4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="angry/angry_none/4.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-290, y=screen_height/2-165)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="티라미수는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_angry5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_angry5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    angry_img = PhotoImage(file="angry/angry_none/5.png")
    img = Label(newWin, image=angry_img)
    img.place(x=screen_width/2-367, y=screen_height/2-350)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="화끈한 우동 불볶이는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none5)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=done)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()


def done():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.geometry("1000x800+480+100")
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(20)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    label1 = Label(newWin, text="죄송합니다.", font=("맑은 고딕", 40))
    label1.pack(side="top", padx=10, pady=10)
    label2 = Label(newWin, text="더 이상 추천드릴 메뉴가 없습니다.", font=("맑은 고딕", 40))
    label2.pack(side="top", padx=10, pady=10)

    btn = Button(newWin, text="끝내기", font=("맑은 고딕", 30), command=exit)
    btn.pack(side="bottom", padx=20, pady=20)
    newWin.mainloop()

def exit():
    root.destroy()


btn_menu1 = Button(root,text="식사", width=25, height=7, font=("휴먼매직체",30), command=open_new_window1)
btn_menu1.place(x=screen_width/6, y=screen_height/6)
btn_menu2 = Button(root,text="음료", width=25, height=7, font=("휴먼매직체",30), command=open_new_window2)
btn_menu2.place(x=screen_width/2, y=screen_height/6)
btn_menu3 = Button(root,text="디저트", width=25, height=7, font=("휴먼매직체",30), command=open_new_window3)
btn_menu3.place(x=screen_width/6, y=screen_height/2)
btn_menu4 = Button(root,text="NONE", width=25, height=7, font=("휴먼매직체",30), command=open_new_window4)
btn_menu4.place(x=screen_width/2, y=screen_height/2)


sketchbook.pack()
sketchbook.create_oval(0, screen_height*0.8, screen_width*0.2, screen_height*1.2, fill = "skyblue" )
sketchbook.create_oval(screen_width*0.2, screen_height*0.8, screen_width*0.4, screen_height*1.2, fill = "skyblue" )
sketchbook.create_oval(screen_width*0.4, screen_height*0.8, screen_width*0.6, screen_height*1.2, fill = "skyblue" )
sketchbook.create_oval(screen_width*0.6, screen_height*0.8, screen_width*0.8,screen_height*1.2, fill = "skyblue" )
sketchbook.create_oval(screen_width*0.8, screen_height*0.8, screen_width,screen_height*1.2, fill = "skyblue" )

root.mainloop()