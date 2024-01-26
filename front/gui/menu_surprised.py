from tkinter import *
from tkinter import filedialog
from os import path


root = Tk()

root.title("surprised")
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
        self.meal_surp1()
    
    def meal_surp1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        _img = PhotoImage(file="surprised/surprised_meal/1.png")
        img = Label(self, image=_img)
        img.place(x=screen_width/2-295, y=screen_height/2-277)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="밀양식 돼지국밥은 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18),  width=20, height=3,bg="skyblue", command=meal_surp2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()

class NewWindow2(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.drink_surp1()
    
    def drink_surp1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        drink_1 = PhotoImage(file="surprised/surprised_drink/1.png")
        label_drink_happy_img = Label(self, image=drink_1)
        label_drink_happy_img.place(x=screen_width/2-148, y=screen_height/2-262)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="할리스 바닐라 딜라이트는 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_surp2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()

class NewWindow3(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.dessert_surp1()
    
    def dessert_surp1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        dessert_1 = PhotoImage(file="surprised/surprised_dessert/1.png")
        img = Label(self, image=dessert_1)
        img.place(x=screen_width/2-268, y=screen_height/2-289)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="밀크티 젤리는 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")
    
        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_surp2)
        btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
        self.mainloop()


class NewWindow4(Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("new")
        self.attributes('-fullscreen', True)
        self.resizable(False, False)
        self.none_surp1()
    
    def none_surp1(self):
        photo_gs = PhotoImage(file="gs25.png").subsample(5)
        label_gs = Label(self, image=photo_gs)
        label_gs.place(x=10, y=20)

        none = PhotoImage(file="surprised/surprised_none/1.png")
        img = Label(self, image=none)
        img.place(x=screen_width/2-217, y=screen_height/2-290)

        l = Label(self, text=" ", font=("휴먼매직체", 50))
        l.pack(side="top")
        q = Label(self, text="꼬북칩 초코츄러스 맛은 어떠신가요?", font=("휴먼매직체", 50))
        q.pack(side="top")

        btn1 = Button(self, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none1)
        btn1.place(x=screen_width/4, y=screen_height*(3/4))
        btn2 = Button(self, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_surp2)
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
    _img = PhotoImage(file="surprised/surprised_meal/1.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-295, y=screen_height/2-277)

    label_lo = Label(newWin, text='"라면 코너 옆 즉석식품 코너 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    meal_img = PhotoImage(file="surprised/surprised_meal/2.png")
    label_meal_img = Label(newWin, image=meal_img)
    label_meal_img.place(x=screen_width/2-280, y=screen_height/2-333)

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
    _img = PhotoImage(file="surprised/surprised_meal/3.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-192, y=screen_height/2-259)

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
    _img = PhotoImage(file="surprised/surprised_meal/4.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-206, y=screen_height/2-231)

    label_lo = Label(newWin, text='"오픈 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="surprised/surprised_meal/5.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-110, y=screen_height/2-113)

    label_lo = Label(newWin, text='"통조림 코너 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")
    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)

    newWin.mainloop()

def meal_surp2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    meal_img = PhotoImage(file="surprised/surprised_meal/2.png")
    label_meal_img = Label(newWin, image=meal_img)
    label_meal_img.place(x=screen_width/2-280, y=screen_height/2-333)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    label_meal_happy2_q = Label(newWin, text="혜자로운 도시락은 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy2_q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_surp3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_surp3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="surprised/surprised_meal/3.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-192, y=screen_height/2-259)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    label_meal_happy3_q = Label(newWin, text="알리오올리오 파스타는 어떠신가요?", font=("휴먼매직체", 50))
    label_meal_happy3_q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_surp4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_surp4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    _img = PhotoImage(file="surprised/surprised_meal/4.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-206, y=screen_height/2-231)
    label_meal_happy4_q = Label(newWin, text="비빔밥은 어떠신가요?",font=("휴먼매직체", 50))
    label_meal_happy4_q.pack(side="top")
    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_meal4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=meal_surp5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def meal_surp5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    _img = PhotoImage(file="surprised/surprised_meal/5.png")
    label_meal_img = Label(newWin, image=_img)
    label_meal_img.place(x=screen_width/2-110, y=screen_height/2-113)

    label_meal_happy5_q = Label(newWin, text="죽은 어떠신가요?", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="surprised/surprised_drink/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-187, y=screen_height/2-135)

    label_lo = Label(newWin, text='"우측 음료 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="surprised/surprised_drink/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-171, y=screen_height/2-210)

    label_lo = Label(newWin, text='"우측 음료 냉장고 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="surprised/surprised_drink/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-243, y=screen_height/2-170)

    label_lo = Label(newWin, text='"우측 오픈 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    drink_img = PhotoImage(file="surprised/surprised_drin/5.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-302, y=screen_height/2-270)

    label_lo = Label(newWin, text='"우측 음료 냉장고 중간"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def drink_surp2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="surprised/surprised_drink/2.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-187, y=screen_height/2-135)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="슈퍼히어로 드링크는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")
    label = Label(newWin, text="<오직 gs25에서만 만나보실 수 있어요!>", font=("맑은 고딕", 30))
    label.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_surp3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_surp3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="surprised/surprised_drink/3.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-171, y=screen_height/2-210)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="제주 녹차는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_surp4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_surp4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="surprised/surprised_drink/4.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-243, y=screen_height/2-170)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="드링킹 요구르트는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_drink4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=drink_surp5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def drink_surp5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    drink_img = PhotoImage(file="surprised/surprised_drin/5.png")
    img = Label(newWin, image=drink_img)
    img.place(x=screen_width/2-302, y=screen_height/2-270)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="복숭아 녹차는 어떠신가요?", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="surprised/surprised_dessert/2.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-320, y=screen_height/2-143)


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
    _img = PhotoImage(file="surprised/surprised_dessert/3.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-253, y=screen_height/2-271)

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
    _img = PhotoImage(file="surprised/surprised_dessert/4.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-142, y=screen_height/2-181)

    label_lo = Label(newWin, text='"가운데 냉장실 과일 코너"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="surprised/surprised_dessert/5.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-158, y=screen_height/2-223)

    label_lo = Label(newWin, text='"과자 코너 하단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def dessert_surp2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="surprised/surprised_dessert/2.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-320, y=screen_height/2-143)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="바이오 요거트는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_surp3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_surp3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="surprised/surprised_dessert/3.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-253, y=screen_height/2-271)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="티라미수 모찌롤은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_surp4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_surp4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="surprised/surprised_dessert/4.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-142, y=screen_height/2-181)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="체리는 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_dessert4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=dessert_surp5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def dessert_surp5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="surprised/surprised_dessert/5.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-158, y=screen_height/2-223)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="카라멜콘 크로아상은 어떠신가요?", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="surprised/surprised_none/2.png")
    img = Label(newWin, image=none)
    img.place(x=screen_height/2-280, y=screen_height/2-333)

    label_lo = Label(newWin, text='"우측 오픈 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="surprised/surprised_none/3.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-346, y=screen_height/2-296)

    label_lo = Label(newWin, text='"라면 코너 옆 즉석식품 코너 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    none = PhotoImage(file="surprised/surprised_none/4.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-180, y=screen_height/2-225)

    label_lo = Label(newWin, text='"과자 코너 옆 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
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
    _img = PhotoImage(file="surprised/surprised_none/5.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-97, y=screen_height/2-221)

    label_lo = Label(newWin, text='"오픈 냉장고 상단"에 있습니다.', fg="blue", font=("휴먼매직체", 50))
    label_lo.pack(side="top")

    out_btn = Button(newWin, text="끝내기", font=("휴먼매직체", 40), command=exit)
    out_btn.pack(side="bottom", padx=50, pady=80)
    newWin.mainloop()



def none_surp2():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="surprised/surprised_none/2.png")
    img = Label(newWin, image=none)
    img.place(x=screen_height/2-280, y=screen_height/2-333)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="커스타드 푸딩은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none2)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_surp3)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_surp3():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="surprised/surprised_none/3.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-346, y=screen_height/2-296)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="갈비 육개장은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none3)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_surp4)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_surp4():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    none = PhotoImage(file="surprised/surprised_none/4.png")
    img = Label(newWin, image=none)
    img.place(x=screen_width/2-180, y=screen_height/2-225)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="초코퐁당 도넛은 어떠신가요?", font=("휴먼매직체", 50))
    q.pack(side="top")

    btn1 = Button(newWin, text="마음에 들어요:)", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=location_none4)
    btn1.place(x=screen_width/4, y=screen_height*(3/4))
    btn2 = Button(newWin, text="다른 추천 :(", font=("휴먼매직체",18), width=20, height=3, bg="skyblue", command=none_surp5)
    btn2.place(x=screen_width*(3/5), y=screen_height*(3/4))
    newWin.mainloop()

def none_surp5():
    newWin = Toplevel(root)
    root.withdraw()
    newWin.attributes('-fullscreen', True)
    newWin.resizable(False, False)
    photo_gs = PhotoImage(file="gs25.png").subsample(5)
    label_gs = Label(newWin, image=photo_gs)
    label_gs.place(x=10, y=20)

    _img = PhotoImage(file="surprised/surprised_none/5.png")
    img = Label(newWin, image=_img)
    img.place(x=screen_width/2-97, y=screen_height/2-221)

    l = Label(newWin, text=" ", font=("휴먼매직체", 50))
    l.pack(side="top")
    q = Label(newWin, text="스크램블&소시기 샌드위치는 어떠신가요?", font=("휴먼매직체", 50))
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