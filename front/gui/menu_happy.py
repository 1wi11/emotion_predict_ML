from tkinter import *
from tkinter import filedialog
from os import path


root = Tk()

root.title("happy")
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
        self.meal_happy1()
    
    def meal_happy1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        meal_happy1 = PhotoImage(file="happy/happy_meal/1.png")
        happy_img = Label(self, image=meal_happy1)
        happy_img.place(x=screen_width/2-315, y=screen_height/2-270)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="슈넬치킨은 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18),  width=20, height=3,bg="skyblue", command=meal_happy2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()

class NewWindow2(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.drink_happy1()
    
    def drink_happy1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        drink_happy1 = PhotoImage(file="happy/happy_drink/1.png")
        label_drink_happy_img = Label(self, image=drink_happy1)
        label_drink_happy_img.place(x=screen_width/2-160, y=screen_height/2-170)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="허쉬초코우유는 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_happy2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()

class NewWindow3(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.dessert_happy1()
    
    def dessert_happy1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        dessert_happy1 = PhotoImage(file="happy/happy_dessert/1.png")
        img = Label(self, image=dessert_happy1)
        img.place(x=screen_width/2-355, y=screen_height/2-165)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="생크림빵은 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        label = Label(self, text="<오직 gs25에서만 만나보실 수 있어요!>", font=("맑은 고딕", 30))
        label.pack(side="top")
    
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_happy2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()


class NewWindow4(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.none_happy1()
    
    def none_happy1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        none = PhotoImage(file="happy/happy_none/1.png")
        happy_img = Label(self, image=none)
        happy_img.place(x=screen_width/2-250, y=screen_height/2-290)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="콰트로치즈 생크림빵은 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        label = Label(self, text="<현재 2+1 행사상품이에요!>", font=("맑은 고딕", 30))
        label.pack(side="top")

        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_happy2)
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
    meal_happy1 = PhotoImage(file="happy/happy_meal/1.png")
    label_meal_happy_img = Label(newWin, image=meal_happy1)
    label_meal_happy_img.place(x=screen_width/2-315, y=screen_height/2-270)
    label_lo = Label(newWin, text='"오픈 냉장고 좌측"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    meal_happy2_img = PhotoImage(file="happy/happy_meal/2.png")
    label_meal_happy2_img = Label(newWin, image=meal_happy2_img)
    label_meal_happy2_img.place(x=screen_width/2-230, y=screen_height/2-205)
    label_lo = Label(newWin, text='"계산대 옆 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    meal_happy3_img = PhotoImage(file="happy/happy_meal/3.png")
    label_meal_happy3_img = Label(newWin, image=meal_happy3_img)
    label_meal_happy3_img.place(x=screen_width/2-235, y=screen_height/2-310)
    label_lo = Label(newWin, text='"계산대 옆 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")
    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def location_meal4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    meal_happy4_img = PhotoImage(file="happy/happy_meal/4.png")
    label_meal_happy4_img = Label(newWin, image=meal_happy4_img)
    label_meal_happy4_img.place(x=screen_width/2-194, y=screen_height/2-192)
    label_lo = Label(newWin, text='"계산대 옆 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    meal_happy5_img = PhotoImage(file="happy/happy_meal/5.png")
    label_meal_happy5_img = Label(newWin, image=meal_happy5_img)
    label_meal_happy5_img.place(x=screen_width/2-210, y=screen_height/2-280)
    label_lo = Label(newWin, text='"오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")
    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()

def meal_happy2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    meal_happy2_img = PhotoImage(file="happy/happy_meal/2.png")
    label_meal_happy2_img = Label(newWin, image=meal_happy2_img)
    label_meal_happy2_img.place(x=screen_width/2-230, y=screen_height/2-205)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    label_meal_happy2_q = Label(newWin, text="이거닭! 강정은 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy2_q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_happy3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_happy3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    meal_happy3_img = PhotoImage(file="happy/happy_meal/3.png")

    label_meal_happy3_img = Label(newWin, image=meal_happy3_img)
    label_meal_happy3_img.place(x=screen_width/2-235, y=screen_height/2-310)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    label_meal_happy3_q = Label(newWin, text="혜자로운 도시락은 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy3_q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_happy4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_happy4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    meal_happy4_img = PhotoImage(file="happy/happy_meal/4.png")
    label_meal_happy4_img = Label(newWin, image=meal_happy4_img)
    label_meal_happy4_img.place(x=screen_width/2-194, y=screen_height/2-192)
    label_meal_happy4_q = Label(newWin, text="불고기 버거는 어떠신가요?",font=("휴먼매직체", 50))
    label_meal_happy4_q.pack(side="top")
    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_happy5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_happy5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    meal_happy5_img = PhotoImage(file="happy/happy_meal/5.png")
    label_meal_happy5_img = Label(newWin, image=meal_happy5_img)
    label_meal_happy5_img.place(x=screen_width/2-210, y=screen_height/2-280)

    label_meal_happy5_q = Label(newWin, text="직화야채곱창볶음은 어떠신가요?", font=("휴먼매직체", 50))
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
    drink_happy1 = PhotoImage(file="happy/happy_drink/1.png")
    label_drink_happy_img = Label(newWin, image=drink_happy1)
    label_drink_happy_img.place(x=screen_width/2-160, y=screen_height/2-170)

    label_lo = Label(newWin, text='"오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_happy2 = PhotoImage(file="happy/happy_drink/2.png")
    label_drink_happy_img = Label(newWin, image=drink_happy2)
    label_drink_happy_img.place(x=screen_width/2-163, y=screen_height/2-255)

    label_lo = Label(newWin, text='"우측 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_happy3_img = PhotoImage(file="happy/happy_drink/3.png")
    img = Label(newWin, image=drink_happy3_img)
    img.place(x=screen_width/2-340, y=screen_height/2-166)

    label_lo = Label(newWin, text='"계산대 우측"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_happy4_img = PhotoImage(file="happy/happy_drink/4.png")
    img = Label(newWin, image=drink_happy4_img)
    img.place(x=screen_width/2-415, y=screen_height/2-215)

    label_lo = Label(newWin, text='"오픈 냉장고 좌측 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_happy5_img = PhotoImage(file="happy/happy_drink/5.png")
    img = Label(newWin, image=drink_happy5_img)
    img.place(x=screen_width/2-108, y=screen_height/2-182)

    label_lo = Label(newWin, text='"우측 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def drink_happy2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_happy2_img = PhotoImage(file="happy/happy_drink/2.png")
    img = Label(newWin, image=drink_happy2_img)
    img.place(x=screen_width/2-163, y=screen_height/2-255)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="모구모구는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_happy3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_happy3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_happy3_img = PhotoImage(file="happy/happy_drink/3.png")
    img = Label(newWin, image=drink_happy3_img)
    img.place(x=screen_width/2-340, y=screen_height/2-166)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="에이드는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_happy4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_happy4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_happy3_img = PhotoImage(file="happy/happy_drink/4.png")
    img = Label(newWin, image=drink_happy3_img)
    img.place(x=screen_width/2-415, y=screen_height/2-215)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="노티드 우유는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_happy5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_happy5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_happy5_img = PhotoImage(file="happy/happy_drink/5.png")
    img = Label(newWin, image=drink_happy5_img)
    img.place(x=screen_width/2-108, y=screen_height/2-182)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="덴마크드링킹은 어떠신가요?", font=("휴먼매직체", 50))
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
    dessert_happy1 = PhotoImage(file="happy/happy_dessert/1.png")
    img = Label(newWin, image=dessert_happy1)
    img.place(x=screen_width/2-355, y=screen_height/2-165)

    label_lo = Label(newWin, text='"오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_happy2_img = PhotoImage(file="happy/happy_dessert/2.png")
    img = Label(newWin, image=drink_happy2_img)
    img.place(x=screen_width/2-335, y=screen_height/2-290)

    label_lo = Label(newWin, text='"과자 코너 오른쪽 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_happy3_img = PhotoImage(file="happy/happy_dessert/3.png")
    img = Label(newWin, image=drink_happy3_img)
    img.place(x=screen_width/2-150, y=screen_height/2-200)

    label_lo = Label(newWin, text='"계산대 앞 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    happy4_img = PhotoImage(file="happy/happy_dessert/4.png")
    img = Label(newWin, image=happy4_img)
    img.place(x=screen_width/2-245, y=screen_height/2-365)

    label_lo = Label(newWin, text='"오픈 냉장고 좌측 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    happy5_img = PhotoImage(file="happy/happy_dessert/5.png")
    img = Label(newWin, image=happy5_img)
    img.place(x=screen_width/2-185, y=screen_height/2-165)

    label_lo = Label(newWin, text='"오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def dessert_happy2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_happy2_img = PhotoImage(file="happy/happy_dessert/2.png")
    img = Label(newWin, image=drink_happy2_img)
    img.place(x=screen_width/2-335, y=screen_height/2-290)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="초코과자는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_happy3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_happy3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_happy3_img = PhotoImage(file="happy/happy_dessert/3.png")
    img = Label(newWin, image=drink_happy3_img)
    img.place(x=screen_width/2-150, y=screen_height/2-200)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="과일 맛 젤리는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_happy4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_happy4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_happy4_img = PhotoImage(file="happy/happy_dessert/4.png")
    img = Label(newWin, image=drink_happy4_img)
    img.place(x=screen_width/2-245, y=screen_height/2-365)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="모찌롤는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_happy5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_happy5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    happy5_img = PhotoImage(file="happy/happy_dessert/5.png")
    img = Label(newWin, image=happy5_img)
    img.place(x=screen_width/2-185, y=screen_height/2-165)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="마카롱은 어떠신가요?", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="happy/happy_none/1.png")
    happy_img = Label(newWin, image=none)
    happy_img.place(x=screen_width/2-250, y=screen_height/2-290)

    label_lo = Label(newWin, text='"오픈 냉장고 좌측 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="happy/happy_none/2.png")
    img = Label(newWin, image=none)
    img.place(x=screen_height/2, y=screen_height/2-305)

    label_lo = Label(newWin, text='"라면 코너 왼쪽 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="happy/happy_none/3.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-165, y=screen_height/2-270)

    label_lo = Label(newWin, text='"과자 코너 옆 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="happy/happy_none/4.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-250, y=screen_height/2-255)

    label_lo = Label(newWin, text='"라면코너 우측 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    happy5_img = PhotoImage(file="happy/happy_none/5.png")
    img = Label(newWin, image=happy5_img)
    img.place(x=screen_width/2-265, y=screen_height/2-243)

    label_lo = Label(newWin, text='"오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def none_happy2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="happy/happy_none/2.png")
    img = Label(newWin, image=none)
    img.place(x=screen_height/2, y=screen_height/2-305)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="탄탄면은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")
    label = Label(newWin, text="<편스토랑 신제품이에요!>", font=("맑은 고딕", 30))
    label.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_happy3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_happy3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="happy/happy_none/3.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-165, y=screen_height/2-270)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="조청 모약과는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")
    label = Label(newWin, text="<현재 2+1 행사상품이에요!>", font=("맑은 고딕", 30))
    label.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_happy4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_happy4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="happy/happy_none/4.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-250, y=screen_height/2-255)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="하얀 짜파게티는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")
    label = Label(newWin, text=("<이번 달 신제품이에요!>"), font=("맑은 고딕", 30))
    label.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_happy5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_happy5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    happy5_img = PhotoImage(file="happy/happy_none/5.png")
    img = Label(newWin, image=happy5_img)
    img.place(x=screen_width/2-265, y=screen_height/2-243)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="생크림도넛은 어떠신가요?", font=("휴먼매직체", 50))
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