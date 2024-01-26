from tkinter import *
from tkinter import filedialog
from os import path


root = Tk()

root.title("neutural")
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
        self.meal_neu1()
    
    def meal_neu1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        _img = PhotoImage(file="neutural/neutural_meal/1.png")
        img = Label(self, image=_img)
        img.place(x=screen_width/2-234, y=screen_height/2-270)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="명란 마요 삼각김밥은 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18),  width=20, height=3,bg="skyblue", command=meal_neu2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()

class NewWindow2(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.drink_neu1()
    
    def drink_neu1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        drink_1 = PhotoImage(file="neutural/neutural_drink/1.png")
        label_drink_happy_img = Label(self, image=drink_1)
        label_drink_happy_img.place(x=screen_width/2-220, y=screen_height/2-150)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="오늘의 과일아채는 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_neu2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()

class NewWindow3(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.dessert_neu1()
    
    def dessert_neu1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        dessert_1 = PhotoImage(file="neutural/neutural_dessert/1.png")
        img = Label(self, image=dessert_1)
        img.place(x=screen_width/2-315, y=screen_height/2-245)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="버터바는 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
    
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_neu2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()


class NewWindow4(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.none_neu1()
    
    def none_neu1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        none = PhotoImage(file="neutural/neutural_none/1.png")
        img = Label(self, image=none)
        img.place(x=screen_width/2-320, y=screen_height/2-242)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="고파두부는 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")

        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_neu2)
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
    _img = PhotoImage(file="neutural/neutural_meal/1.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-234, y=screen_height/2-270)

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
    meal_img = PhotoImage(file="neutural/neutural_meal/2.png")
    label_meal_img = Label(newWin, image=meal_img)
    label_meal_img.place(x=screen_width/2-250, y=screen_height/2-112)

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
    _img = PhotoImage(file="neutural/neutural_meal/3.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-271, y=screen_height/2-270)

    label_lo = Label(newWin, text='"좌측 오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="neutural/neutural_meal/4.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-130, y=screen_height/2-322)

    label_lo = Label(newWin, text='"오픈 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="neutural/neutural_meal/5.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-100, y=screen_height/2-163)

    label_lo = Label(newWin, text='"좌측 오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")
    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)

    newWin.mainloop()

def meal_neu2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    meal_img = PhotoImage(file="neutural/neutural_meal/2.png")
    label_meal_img = Label(newWin, image=meal_img)
    label_meal_img.place(x=screen_width/2-250, y=screen_height/2-112)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    label_meal_happy2_q = Label(newWin, text="페퍼로니 치즈 치아바타는 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy2_q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_neu3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_neu3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="neutural/neutural_meal/3.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-271, y=screen_height/2-270)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    label_meal_happy3_q = Label(newWin, text="죠스 떡볶이는 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy3_q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_neu4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_neu4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    _img = PhotoImage(file="neutural/neutural_meal/4.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-130, y=screen_height/2-322)
    label_meal_happy4_q = Label(newWin, text="황치즈 샌드위치는 어떠신가요?",font=("휴먼매직체", 50))
    label_meal_happy4_q.pack(side="top")
    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_neu5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_neu5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    _img = PhotoImage(file="neutural/neutural_meal/5.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-100, y=screen_height/2-163)

    label_meal_happy5_q = Label(newWin, text="왕교자 그라탕은 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy5_q.pack(side="top")
    label = Label(newWin, text="<오직 gs25에서만 만나보실 수 있어요!>", font=("맑은 고딕", 30))
    label.pack(side="top")

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
    drink_1 = PhotoImage(file="surprised/surprised_drink/1.png")
    label_drink_happy_img = Label(newWin, image=drink_1)
    label_drink_happy_img.place(x=screen_width/2-148, y=screen_height/2-262)


    label_lo = Label(newWin, text='"우측 오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="neutural/neutural_drink/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-216, y=screen_height/2-242)

    label_lo = Label(newWin, text='"우측 음료 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="neutural/neutural_drink/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-146, y=screen_height/2-178)

    label_lo = Label(newWin, text='"우측 오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="neutural/neutural_drink/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-180, y=screen_height/2-291)

    label_lo = Label(newWin, text='"캔음료 코너 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="neutural/neutural_drink/5.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-188, y=screen_height/2-112)

    label_lo = Label(newWin, text='"우측 오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def drink_neu2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="neutural/neutural_drink/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-216, y=screen_height/2-242)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="아메리카노 / 라떼는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_neu3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_neu3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="neutural/neutural_drink/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-146, y=screen_height/2-178)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="대용량 아이스티는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_neu4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_neu4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="neutural/neutural_drink/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-180, y=screen_height/2-291)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="코카콜라 제로 레몬 맛은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_neu5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_neu5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="neutural/neutural_drink/5.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-188, y=screen_height/2-112)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="강릉커피는 어떠신가요?", font=("휴먼매직체", 50))
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
    dessert_1 = PhotoImage(file="surprised/surprised_dessert/1.png")
    img = Label(newWin, image=dessert_1)
    img.place(x=screen_width/2-268, y=screen_height/2-289)
    
    label_lo = Label(newWin, text='"우측 오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="neutural/neutural_dessert/2.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-311, y=screen_height/2-320)


    label_lo = Label(newWin, text='"오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="neutural/neutural_dessert/3.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-208, y=screen_height/2-313)

    label_lo = Label(newWin, text='"오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="neutural/neutural_dessert/4.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-178, y=screen_height/2-205)

    label_lo = Label(newWin, text='"오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="neutural/neutural_dessert/5.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-243, y=screen_height/2-208)

    label_lo = Label(newWin, text='"과자 코너 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def dessert_neu2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="neutural/neutural_dessert/2.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-311, y=screen_height/2-320)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="바스크치즈케익은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_neu3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_neu3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="neutural/neutural_dessert/3.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-208, y=screen_height/2-313)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="망고쏙우유찹쌀떡은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_neu4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_neu4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="neutural/neutural_dessert/4.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-178, y=screen_height/2-205)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="딸기케이크는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_neu5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_neu5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="neutural/neutural_dessert/5.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-243, y=screen_height/2-208)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="꼬북칩 인절미 맛은 어떠신가요?", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="surprised/surprised_none/1.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-217, y=screen_height/2-290)
    
    label_lo = Label(newWin, text='"과자 코너 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="neutural/neutural_none/2.png")
    img = Label(newWin, image=none)
    img.place(x=screen_height/2-158, y=screen_height/2-208)

    label_lo = Label(newWin, text='"오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="neutural/neutural_none/3.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-300, y=screen_height/2-235)

    label_lo = Label(newWin, text='"좌측 오픈 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="neutural/neutural_none/4.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-140, y=screen_height/2-141)

    label_lo = Label(newWin, text='"오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="neutural/neutural_none/5.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-321, y=screen_height/2-353)

    label_lo = Label(newWin, text='"즉석식품 코너 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def none_neu2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="neutural/neutural_none/2.png")
    img = Label(newWin, image=none)
    img.place(x=screen_height/2-158, y=screen_height/2-208)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="행운약과 컵케이크는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")
    label = Label(newWin, text="<오직 gs25에서만 만나보실 수 있어요!>", font=("맑은 고딕", 30))
    label.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_neu3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_neu3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="neutural/neutural_none/3.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-300, y=screen_height/2-235)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="스모크 닭다리는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")
    label = Label(newWin, text="<오직 gs25에서만 만나보실 수 있어요!>", font=("맑은 고딕", 30))
    label.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_neu4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_neu4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="neutural/neutural_none/4.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-140, y=screen_height/2-141)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="카이막 그릭요거트는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")
    label = Label(newWin, text="<오직 gs25에서만 만나보실 수 있어요!>", font=("맑은 고딕", 30))
    label.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_neu5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_neu5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="neutural/neutural_none/5.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-321, y=screen_height/2-353)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="마라량피는 어떠신가요?", font=("휴먼매직체", 50))
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