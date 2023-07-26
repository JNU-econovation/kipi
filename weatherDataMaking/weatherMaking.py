import tkinter as tk
import tkinter.ttk as ttk
from PIL import ImageTk, Image
import pandas as pd
import json

NAME = str()
IMG_SIZE = (400,600)

all_df = pd.read_csv('data/all_data.csv')
part_df = pd.read_csv('data/part_by.csv')

cnt_by_NAME = {
    'gunhyeong' : 0,
    'seoha' : 0,
    'chaeseong' : 0,
    'suhyeon' : 0
}

try:
    with open('data/cnt_by_name.json', 'r') as f:
        cnt_by_NAME = json.load(f)
except:
    pass

window = tk.Tk()
window.title('weather data maker')
window.geometry("700x700+100+100")
# window.resizable(False, False)


page1 = tk.Frame(window, width=700, height=700, background='white')
page1.grid(row=0,column=0)



def insertName(event):
    global NAME

    input_to_name = {
        '건형' : 'gunhyeong',
        '김건형' : 'gunhyeong',
        'gunhyeong' : 'gunhyeong',
        'kimgunhyeong' : 'gunhyeong',

        '서하' : 'seoha',
        '김서하' : 'seoha',
        'seoha' : 'seoha',
        'kimseoha' : 'seoha',

        '채승' : 'chaeseong',
        '임채승' : 'chaeseong',
        'chaeseong' : 'chaeseong',
        'imchaeseong' : 'chaeseong',

        '수현' : 'suhyeon',
        '박수현' : 'suhyeon',
        'suhyeon' : 'suhyeon',
        'parksuhyeon' : 'suhyeon'
    }

    s = input_to_name.get(name_input.get().replace(' ', ''))

    if s == None:
        return
    else:
        NAME = s
        page1.destroy()
        page2.grid(row=0,column=0)
        nxt_pic()


name_input = tk.Entry(page1)
name_input.bind("<Return>", insertName)
name_input.grid(row=0,column=0,)

page2 = tk.Frame(window, width=700, height=700)



def exit_program():
    window.destroy()

def imagePrepro(dir):
    global tk_img

    img = Image.open(dir)
    new_size = list(IMG_SIZE)
    if img.size[0]*3/2 > img.size[1]:
        ratio = IMG_SIZE[0]/img.size[0]
        new_size[1] = int(img.size[1]*ratio)
    else :
        ratio = IMG_SIZE[1]/img.size[1]
        new_size[0] = int(img.size[0]*ratio)

    
    img_resize = img.resize(new_size, Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(img_resize)
    return tk_img



tk_img = imagePrepro('img/white.jpg')

img_frame = tk.Label(page2, width=IMG_SIZE[0], height=IMG_SIZE[1], image=tk_img)
img_frame.grid(row=0, column=0)


def pre_pic():
    global tk_img
    global cnt_by_NAME

    cnt_by_NAME[NAME] -= 1
    tk_img = imagePrepro('img/clothes/'+str(part_df[NAME][cnt_by_NAME[NAME]-1])+'.jpg')
    img_frame.configure(image=tk_img)
    

def nxt_pic():
    global tk_img
    global cnt_by_NAME

    tk_img = imagePrepro('img/clothes/'+str(part_df[NAME][cnt_by_NAME[NAME]])+'.jpg')
    img_frame.configure(image=tk_img)
    pertage.configure(text=f"{cnt_by_NAME[NAME]}/2921\n {round(cnt_by_NAME[NAME]*100/2921,4)}%")
    cnt_by_NAME[NAME] += 1

def insertWeather(event):
    s = entry.get()
    if s == '':
        return
    elif not s.isdecimal() :
        return
    elif int(s) < 1 or int(s) > 12:
        return
    else:
        all_df.loc[all_df['data_num']==part_df[NAME][cnt_by_NAME[NAME]]] = s
        all_df.to_csv('all_data.csv', index=False)
        with open('data/cnt_by_name.json', 'w', encoding='utf-8') as f:
            json.dump(cnt_by_NAME, f, indent=4)
        pre_input.configure(text=entry.get())
        entry.delete(0,len(entry.get()))
        nxt_pic()



right_box = tk.LabelFrame(page2, borderwidth=0)
right_box.grid(row=0,column=1)

treeview = ttk.Treeview(right_box, 
                        columns=["weather","temperature(℃)"],
                        displaycolumns=["weather","temperature(℃)"],
                        height=15)

treeview.column("#1", width=100, anchor="center")
treeview.heading("#1", text="weather")
treeview.column("#2", width = 100)
treeview.heading("#2", text="temperature(℃)")
treeview["show"] = "headings"

treelist = [(1, "over 33"), 
            (2, "33~28"),
            (3, "28~25"),
            (4, "25~23"),
            (5, "23~21"),
            (6, "21~18"),
            (7, "18~15"),
            (8, "15~13"),
            (9, "13~9"),
            (10, "9~6"),
            (11, "6~0"),
            (12, "under 0")]

for i in range(len(treelist)):
    treeview.insert('', 'end', text='', values=treelist[i], iid=i)

treeview.grid(row=0, column = 0, padx=10)

# label1 = tk.Label(window,text='weather      temperature(℃)       humidity(%)').grid(row=0, column=1)


label_frame = tk.LabelFrame(right_box, borderwidth=0)
label_frame.grid(row=1,column=0, pady=30)

pre_input_noti = tk.Label(label_frame, text='이전 입력 : ', anchor='w')
pre_input_noti.grid(row=2, column=0)

pre_input = tk.Label(label_frame, text='없음', anchor='w')
pre_input.grid(row=2, column=1)

entry = tk.Entry(label_frame)
entry.bind("<Return>", insertWeather)
entry.grid(row=0,column=0,)

btn_frame = tk.LabelFrame(label_frame,borderwidth=0)
btn_frame.grid(row=1, column=0, pady=10)

pre_btn = tk.Button(btn_frame,text='before',bg='blue',command=pre_pic).grid(row=0,column=1)
nxt_btn = tk.Button(btn_frame,text='after',bg='blue',command=nxt_pic).grid(row=0,column=2)
exit_btn = tk.Button(btn_frame,text="exit",bg='red',command=exit_program).grid(row=0, column=3)

pertage = tk.Label(btn_frame, text=f"0/2921\n 0%", anchor = 'w')
pertage.grid(row=1, column=0, columnspan=3)

window.mainloop()