from tkinter import *
from tkinter import filedialog
from os import path


root = Tk()

root.title("sad")
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
        self.meal_sad1()
    
    def meal_sad1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        _img = PhotoImage(file="sad/sad_meal/1.png")
        img = Label(self, image=_img)
        img.place(x=screen_width/2-255, y=screen_height/2-304)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="명란 크림 파스타는 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18),  width=20, height=3,bg="skyblue", command=meal_sad2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()

class NewWindow2(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.drink_sad1()
    
    def drink_sad1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        drink_1 = PhotoImage(file="sad/sad_drink/1.png")
        label_drink_happy_img = Label(self, image=drink_1)
        label_drink_happy_img.place(x=screen_width/2-187, y=screen_height/2-239)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="바나나맛 우유는 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_sad2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()

class NewWindow3(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.dessert_sad1()
    
    def dessert_sad1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        dessert_1 = PhotoImage(file="sad/sad_dessert/1.png")
        img = Label(self, image=dessert_1)
        img.place(x=screen_width/2-165, y=screen_height/2-190)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="벤&제리 아이스크림은 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
    
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_sad2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()


class NewWindow4(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.none_sad1()
    
    def none_sad1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        none = PhotoImage(file="sad/sad_none/1.png")
        img = Label(self, image=none)
        img.place(x=screen_width/2-172, y=screen_height/2-183)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="납작복숭아 생크림빵은 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")

        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_sad2)
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
    _img = PhotoImage(file="sad/sad_meal/1.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-255, y=screen_height/2-304)

    label_lo = Label(newWin, text='"좌측 오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    meal_img = PhotoImage(file="sad/sad_meal/2.png")
    label_meal_happy2_img = Label(newWin, image=meal_img)
    label_meal_happy2_img.place(x=screen_width/2-294, y=screen_height/2-233)

    label_lo = Label(newWin, text='"오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="sad/sad_meal/3.png")
    label_meal_happy3_img = Label(newWin, image=_img)
    label_meal_happy3_img.place(x=screen_width/2-248, y=screen_height/2-233)

    label_lo = Label(newWin, text='"라면 코너 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="sad/sad_meal/4.png")
    label_meal_happy4_img = Label(newWin, image=_img)
    label_meal_happy4_img.place(x=screen_width/2-232, y=screen_height/2-173)

    label_lo = Label(newWin, text='"좌측 오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="sad/sad_meal/5.png")
    label_meal_happy5_img = Label(newWin, image=_img)
    label_meal_happy5_img.place(x=screen_width/2-210, y=screen_height/2-226)

    label_lo = Label(newWin, text='"라면 코너 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")
    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)

    newWin.mainloop()

def meal_sad2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    meal_img = PhotoImage(file="sad/sad_meal/2.png")
    label_meal_happy2_img = Label(newWin, image=meal_img)
    label_meal_happy2_img.place(x=screen_width/2-294, y=screen_height/2-233)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    label_meal_happy2_q = Label(newWin, text="빅&더블 버거는 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy2_q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_sad3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_sad3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="sad/sad_meal/3.png")
    label_meal_happy3_img = Label(newWin, image=_img)
    label_meal_happy3_img.place(x=screen_width/2-248, y=screen_height/2-233)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    label_meal_happy3_q = Label(newWin, text="로제 불닭볶음면은 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy3_q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_sad4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_sad4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    _img = PhotoImage(file="sad/sad_meal/4.png")
    label_meal_happy4_img = Label(newWin, image=_img)
    label_meal_happy4_img.place(x=screen_width/2-232, y=screen_height/2-173)
    label_meal_happy4_q = Label(newWin, text="쌈채소 석쇠불고기 도시락은 어떠신가요?",font=("휴먼매직체", 50))
    label_meal_happy4_q.pack(side="top")
    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_sad5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_sad5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    _img = PhotoImage(file="sad/sad_meal/5.png")
    label_meal_happy5_img = Label(newWin, image=_img)
    label_meal_happy5_img.place(x=screen_width/2-210, y=screen_height/2-226)

    label_meal_happy5_q = Label(newWin, text="돈코츠향 라멘은 어떠신가요?", font=("휴먼매직체", 50))
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
    drink_1 = PhotoImage(file="sad/sad_drink/1.png")
    label_drink_happy_img = Label(newWin, image=drink_1)
    label_drink_happy_img.place(x=screen_width/2-187, y=screen_height/2-239)


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
    drink_img = PhotoImage(file="sad/sad_drink/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-209, y=screen_height/2-340)

    label_lo = Label(newWin, text='"우측 오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="sad/sad_drink/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-155, y=screen_height/2-155)

    label_lo = Label(newWin, text='"우측 오픈 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="sad/sad_drink/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-185, y=screen_height/2-355)

    label_lo = Label(newWin, text='"우측 오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="sad/sad_drink/5.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-165, y=screen_height/2-250)

    label_lo = Label(newWin, text='"우측 오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def drink_sad2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="sad/sad_drink/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-209, y=screen_height/2-340)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="연유라떼는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_sad3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_sad3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="sad/sad_drink/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-155, y=screen_height/2-155)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="타이거슈가 밀크티는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_sad4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_sad4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="sad/sad_drink/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-185, y=screen_height/2-355)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="버터커피는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_sad5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_sad5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="sad/sad_drink/5.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-165, y=screen_height/2-250)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="리얼 과일우유는 어떠신가요?", font=("휴먼매직체", 50))
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
    dessert_1 = PhotoImage(file="sad/sad_dessert/1.png")
    img = Label(newWin, image=dessert_1)
    img.place(x=screen_width/2-165, y=screen_height/2-190)

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
    drink_img = PhotoImage(file="sad/sad_dessert/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-275, y=screen_height/2-286)


    label_lo = Label(newWin, text='"오픈 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="sad/sad_dessert/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-192, y=screen_height/2-307)

    label_lo = Label(newWin, text='"아이스크림 코너 우측"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="sad/sad_dessert/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-145, y=screen_height/2-349)

    label_lo = Label(newWin, text='"오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    sad_img = PhotoImage(file="sad/sad_dessert/5.png")
    img = Label(newWin, image=sad_img)
    img.place(x=screen_width/2-405, y=screen_height/2-208)

    label_lo = Label(newWin, text='"오픈 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def dessert_sad2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="sad/sad_dessert/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-275, y=screen_height/2-286)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="제주말차 딸기 샌드위치는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_sad3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_sad3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="sad/sad_dessert/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-192, y=screen_height/2-307)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="노티드 아이스크림은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_sad4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_sad4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="sad/sad_dessert/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-145, y=screen_height/2-349)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="허쉬 쿠앤크 모찌롤은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_sad5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_sad5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    sad_img = PhotoImage(file="sad/sad_dessert/5.png")
    img = Label(newWin, image=sad_img)
    img.place(x=screen_width/2-405, y=screen_height/2-208)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="슈퍼말차 티라미수는 어떠신가요?", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="sad/sad_none/1.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-172, y=screen_height/2-183)
    
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
    none = PhotoImage(file="sad/sad_none/2.png")
    img = Label(newWin, image=none)
    img.place(x=screen_height/2-272, y=screen_height/2-204)

    label_lo = Label(newWin, text='"좌측 오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="sad/sad_none/3.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-223, y=screen_height/2-270)

    label_lo = Label(newWin, text='"오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="sad/sad_none/4.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-290, y=screen_height/2-182)

    label_lo = Label(newWin, text='"오픈 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    sad_img = PhotoImage(file="sad/sad_none/5.png")
    img = Label(newWin, image=sad_img)
    img.place(x=screen_width/2-145, y=screen_height/2-345)

    label_lo = Label(newWin, text='"우측 오픈 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def none_sad2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="sad/sad_none/2.png")
    img = Label(newWin, image=none)
    img.place(x=screen_height/2-272, y=screen_height/2-204)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="짜장밥 삼각김밥은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")
    label = Label(newWin, text="<편스토랑 제품이에요!>", font=("맑은 고딕", 30))
    label.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_sad3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_sad3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="sad/sad_none/3.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-223, y=screen_height/2-270)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="오페라 케이크는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_sad4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_sad4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="sad/sad_none/4.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-290, y=screen_height/2-182)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="아우어 인절미는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_sad5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_sad5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    sad_img = PhotoImage(file="sad/sad_none/5.png")
    img = Label(newWin, image=sad_img)
    img.place(x=screen_width/2-145, y=screen_height/2-345)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="직화 통삼겹은 어떠신가요?", font=("휴먼매직체", 50))
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