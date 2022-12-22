from tkinter import filedialog as fd
from tensorflow import keras
import customtkinter as ct
import pandas as pd
import numpy as np
from tkinter import *
import re
import os

path_to_model = "model3.h5"

ct.set_appearance_mode("System")
ct.set_default_color_theme("dark-blue")
app = ct.CTk()
app.geometry("605x425")
app.title("Diploma № 1")

def Path():
    global path_to_model
    path_to_model = fd.askopenfilename(filetypes=[("Neural models", ".h5")])
    label_model['text']=path_to_model
def generate_dataset(df):
    tensorlist = []
    labelist = []
    maxabs = 0
    for row in df.values:

        query = row[0]
        label = row[1]

        url_part = [ord(x) if (ord(x) < 129) else 0 for x in query[:784]]
        url_part += [0] * (784 - len(url_part))

        maxim = max(url_part)
        if maxim > maxabs:
            maxabs = maxim
        x = np.array(url_part).reshape(28, 28)

        if label == 1:
            y = np.array([0, 1], dtype=np.int8)
        else:
            y = np.array([1, 0], dtype=np.int8)
        tensorlist.append(x)
        labelist.append(y)

    return tensorlist, labelist
def work():
    global path_to_model

    post = ''
    textfile = open('C:\\Apache24\\logs\\error.log', 'r')
    filetext = textfile.read()
    textfile.close()

    matches_1 = re.findall("\[client (.*):[0-9]*] .* POST", filetext)
    matches_2 = re.findall("csrfmiddlewaretoken=.*&text=(.*)", filetext)

    print(matches_1)

    for x in range(len(matches_2)):
        print(matches_2[x])
        matches_2[x] = matches_2[x].replace('%3C', '<')
        matches_2[x] = matches_2[x].replace('%3E', '>')
        matches_2[x] = matches_2[x].replace('%3F', '?')
        matches_2[x] = matches_2[x].replace('%21', '!')
        matches_2[x] = matches_2[x].replace('%28', '(')
        matches_2[x] = matches_2[x].replace('%29', ')')
        matches_2[x] = matches_2[x].replace('%2F', '/')

        if "+" in matches_2[x]:
            matches_2[x] = matches_2[x].replace('+', ' ')



        post += f'{matches_2[x]}, {matches_1[x]}\n'

    with open('sample.txt', 'w') as the_file:
        for x in matches_2:
            the_file.write(f'{x}\n')

    with open('sample_id.txt', 'w') as the_file:
        for x in matches_1:
            the_file.write(f'{x}\n')

    print(matches_2)

    checker,blocklist = [],[]
    print(path_to_model)
    model = keras.models.load_model(path_to_model)

    f = open('C:\\Users\\alexp\\PycharmProjects\\Diplom\\Vul\\sample.txt', encoding='utf-8', mode='r')
    allqueries = f.readlines()
    f.close()

    allqueriespd = pd.DataFrame({'query': allqueries, 'label': [2 for x in allqueries]})
    print(allqueriespd)

    X, y = generate_dataset(allqueriespd)

    prediction = model.predict(np.array(X)/128.0)

    pred = prediction.tolist()

    print(pred)

    amount = 0

    for x in range(len(pred)):
        if pred[x][0] > 0.5:
            amount += 1
            checker.append(0)
        else:
            checker.append(1)

    print(checker)

    with open('C:\\Users\\alexp\\PycharmProjects\\Diplom\\Vul\\sample_id.txt') as f:
        lines = f.read().splitlines()

    print(lines)

    for x in range(len(lines)):
        if checker[x] == 1:
            blocklist.append(lines[x])

    blocklist = list(dict.fromkeys(blocklist))

    print(blocklist)

    with open('C:\\Apache24\\conf\\IPList.conf', 'w') as the_file:
        for x in blocklist:
            the_file.write(f'Require not ip {x}\n')

    for x in pred:
        post += f"{str(x)}\n"

    post += f'Amount of normal requests {amount}\n'
    post += f"Amount of abnormal requests {len(pred)- amount}\n"

    post += f"Blacklist: {str(blocklist)}"

    T.insert(END, post)

    os.system('net stop Apache24 && net start Apache24')

button_check = ct.CTkButton(master=app,width=120,height=32,border_width=0,corner_radius=5,text="Check",fg_color="#FF5F5F",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 10),command=work)
button_model = ct.CTkButton(master=app,width=120,height=32,border_width=0,corner_radius=5,text="Choose the model",fg_color="#FF5F5F",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 10),command=Path)
label_model = ct.CTkLabel(master=app,text=path_to_model,text_color = "#FCFCFC",text_font=("Bauhaus 93", 10))
T = Text(app, height=9, width=45, font=("Verdana", 14))

button_check.place(x=145, y=20)
button_model.place(x=345, y=20)
label_model.place(x= 150, y =90)
T.place(x=30, y=180)

app.mainloop()
