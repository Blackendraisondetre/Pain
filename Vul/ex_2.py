import customtkinter as ct
import pandas as pd
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from mlxtend.plotting import plot_confusion_matrix
from keras import layers
from tensorflow import keras
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog as fd

ct.set_appearance_mode("System")
ct.set_default_color_theme("dark-blue")
app = ct.CTk()
app.geometry("370x185")
app.title("Diploma № 2")

#Test
path_to_model_test = "model3.h5"
path_to_sample_test = "sample.txt"

#Train
path_to_sample_bening_train = "Bening.txt"
path_to_sample_malicious_train = "Xss.txt"
path_to_model_train = "model3.h5"



chel_train = IntVar()
chel_train.set(1)

rad_train = IntVar()
rad_train.set(0)

check_mod_acc = IntVar()
check_mod_acc.set(0)
check_mod_loss = IntVar()
check_mod_loss.set(0)
check_conf_mat = IntVar()
check_conf_mat.set(0)


def radiobutton_event():
    if rad_train.get() ==2:
        train_model_button.place(x=50,y=300)
        train_model_label.place(x=325, y=300+20)

    elif rad_train.get() == 1:
        train_model_button.place_forget()
        train_model_label.place_forget()

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

def bummer_test():

    app.geometry("800x850")
    button_train.place_forget()
    button_test.place_forget()
    button_exit.place_forget()
    button_main_meun_test.place(x=320, y=750)

    test_model_button.place(x = 50, y = 50)
    test_sample_button.place(x = 50, y = 150)

    test_model_label.place(x = 325, y = 50+20)
    test_sample_label.place(x = 325, y = 150+20)

    button_test_start.place(x = 505, y = 100)

    Test.place(x=30, y=250)

def bummer_train():

    app.geometry("820x810")
    button_train.place_forget()
    button_test.place_forget()
    button_exit.place_forget()
    button_main_meun_train.place(x=50, y=730)

    train_bening_sample_button.place(x=50, y=50)
    train_malicious_sample_button.place(x=50, y=150)

    train_bening_sample_label.place(x=325, y=50 + 20)
    train_malicious_sample_label.place(x=325, y=150 + 20)

    radiobutton_1.place(x = 50, y = 245)
    radiobutton_2.place(x = 195, y = 245)

    save_train.place(x =50, y=400)
    save_train.toggle()

    check_model_accuracy.place(x =50, y =450)
    check_model_accuracy.toggle()

    check_model_loss.place(x =50, y =500)
    check_model_loss.toggle()

    check_confusion_matrix.place(x =50, y =550)
    check_confusion_matrix.toggle()

    epochs_entry.place(x = 50, y = 600)

    train_button_start.place(x=50, y = 650)

    Train.place(x=320, y=400)

def main_menu_test():

    app.geometry("370x185")
    button_train.place(x=50, y=20)
    button_test.place(x=200, y=20)
    button_exit.place(x=50, y=100)

    test_model_button.place_forget()
    test_sample_button.place_forget()
    test_model_label.place_forget()
    test_sample_label.place_forget()
    button_test_start.place_forget()
    button_main_meun_test.place_forget()

    Test.place_forget()

def main_menu_train():

    app.geometry("370x185")
    button_train.place(x=50, y=20)
    button_test.place(x=200, y=20)
    button_exit.place(x=50, y=100)

    train_bening_sample_button.place_forget()
    train_malicious_sample_button.place_forget()
    train_bening_sample_label.place_forget()
    train_malicious_sample_label.place_forget()
    button_main_meun_train.place_forget()
    radiobutton_1.place_forget()
    radiobutton_2.place_forget()
    save_train.place_forget()
    check_model_accuracy.place_forget()
    check_model_loss.place_forget()
    check_confusion_matrix.place_forget()
    epochs_entry.place_forget()
    train_button_start.place_forget()
    Train.place_forget()



    #button_test_start.place_forget()
    #Test.place_forget()

def Path_model_test():
    global path_to_model_test
    path_to_model_test = fd.askopenfilename(filetypes=[("Neural models", ".h5")])
    test_model_label['text']=path_to_model_test

def Path_sample_test():
    global path_to_sample_test
    path_to_sample_test = fd.askopenfilename(filetypes=[("Samples", ".txt")])
    test_sample_label['text'] = path_to_sample_test

#Train

def Path_sample_bening_train():
    global path_to_sample_bening_train
    path_to_sample_bening_train = fd.askopenfilename(filetypes=[("Samples", ".txt")])
    train_bening_sample_label['text'] = path_to_sample_bening_train

def Path_sample_malicious_train():
    global path_to_sample_malicious_train
    path_to_sample_malicious_train = fd.askopenfilename(filetypes=[("Samples", ".txt")])
    train_malicious_sample_label['text'] = path_to_sample_malicious_train

def Path_model_train():
    global path_to_model_test
    path_to_model_test = fd.askopenfilename(filetypes=[("Neural models", ".h5")])
    train_model_label['text']=path_to_model_test

def show_train():
    lol = chel_train.get()
    print(chel_train.get())
    if chel_train.get() == 0:
        save_entry.place(x=170, y=400)
    elif chel_train.get() == 1:
        try:
            save_entry.place_forget()
        except:
            print("lol")

def plot_graph(history,indexes):

    if indexes[0] ==1:
        plt.plot(history.history["accuracy"])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc="upper left")
        plt.show()

    if indexes[1]==1:
        plt.plot(history.history['loss'])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()



def train_start():
    global model


    print(path_to_sample_malicious_train)
    f = open(path_to_sample_malicious_train, encoding='utf-8', mode='r')
    badqueries = f.readlines()
    f.close()

    f = open(path_to_sample_bening_train, encoding='utf-8', mode='r')
    goodqueries = f.readlines()
    f.close()

    badqueriespd = pd.DataFrame({'query': badqueries,'label': [1 for x in badqueries]})
    goodqueries = pd.DataFrame({'query': goodqueries,'label': [0 for x in goodqueries]})
    allqueries = pd.concat([badqueriespd,goodqueries])

    X,y = generate_dataset(allqueries)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if rad_train.get() == 2:

        model = Sequential()
        model.add(layers.Input(shape=(28,28,1)))
        model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(MaxPool2D((3,3)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',metrics.Recall(),metrics.Precision()])

    else:
        model = keras.models.load_model(path_to_model_test)


    print(len(X_train))
    print(len(y_train))
    print(model.summary)

    amount_of_epochs = int(epochs_entry.get())

    history = model.fit(np.array(X_train)/128.0,np.array(y_train),epochs=amount_of_epochs,batch_size=100)

    train_ans = ''

    print(history.history['accuracy'])

    for x in range(len(history.history["accuracy"])):
        train_ans += (f"Epoch № {x + 1} : {round(history.history['accuracy'][x], 4)}\n")


    Train.delete('1.0', END)
    Train.insert(END, '=======List of accuracy per epoch=======\n')
    Train.insert(END, train_ans)

    plot_graph(history, [check_mod_acc.get(),check_mod_loss.get()])

    if check_conf_mat.get() ==1:
        y_pred = model.predict(np.array(X_test) / 128.0)
        classes_y = np.argmax(y_pred, axis=1)

        new_y = []
        pain = np.array(y_test).tolist()
        for x in pain:
            if x[0] == 1:
                new_y.append(0)
            else:
                new_y.append(1)

        cm = confusion_matrix(np.array(new_y), classes_y)
        print(cm)
        plot_confusion_matrix(conf_mat=cm)
        plt.show()

        print('aee')

    print(f"Model accuracy {check_mod_acc.get()}")
    print(f"Model loss {check_mod_loss.get()}")
    print(f"Model matrix {check_conf_mat.get()}")

    #weights = model.get_weights()
    #print(weights)

    print(chel_train.get())

    if chel_train.get() == 1:
        print("model saved")
        new_mod_name = save_entry.get()
        model.save(new_mod_name)


def test_start():

    model = keras.models.load_model(path_to_model_test)
    f = open(path_to_sample_test, encoding='utf-8', mode='r')
    allqueries = f.readlines()
    f.close()
    allqueriespd = pd.DataFrame({'query': allqueries, 'label': [2 for x in allqueries]})
    print(allqueriespd)

    X, y = generate_dataset(allqueriespd)

    prediction = model.predict(np.array(X) / 128.0)

    pred = prediction.tolist()

    print(pred)

    amount = 0

    for x in range(len(pred)):
        if pred[x][0] > 0.5:
            amount += 1

    test_ans = ''

    for x in range(len(pred)):
        test_ans += (f"№{x+1} : {allqueriespd['query'][x]} {[round(num,4) for num in pred[x]]}\n")

    print(f'Amount of normal requests {amount}\n')
    print(f"Amount of abnormal requests {len(pred) - amount}\n")

    Test.delete('1.0', END)
    Test.insert(END,'=================Amount of requests================\n')
    Test.insert(END, f'Amount of normal requests {amount}\n')
    Test.insert(END, f'Amount of abnormal requests {len(pred) - amount}\n')
    Test.insert(END,'==================List of samples==================\n')
    Test.insert(END, test_ans)

def exit():
    app.quit()


button_train = ct.CTkButton(master=app,width=120,height=32*2,border_width=0,corner_radius=5,text="Train",fg_color="#FF5F5F",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 20),command=bummer_train)
button_test = ct.CTkButton(master=app,width=120,height=32*2,border_width=0,corner_radius=5,text="Test",fg_color="#FF5F5F",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 20),command=bummer_test)
button_exit = ct.CTkButton(master=app,width=270,height=32*2,border_width=0,corner_radius=5,text="Exit",fg_color="#FCFCFC",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 20),command=exit)

#Test

button_main_meun_test = ct.CTkButton(master=app,width=120,height=32*2,border_width=0,corner_radius=5,text="Main Menu",fg_color="#FCFCFC",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 20),command=main_menu_test)

test_model_button = ct.CTkButton(master=app,width=250,height=32*2,border_width=0,corner_radius=5,text="Choose the model",fg_color="#FF5F5F",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 16),command=Path_model_test)
test_sample_button = ct.CTkButton(master=app,width=250,height=32*2,border_width=0,corner_radius=5,text="Choose the samples",fg_color="#FF5F5F",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 16),command=Path_sample_test)

test_model_label = ct.CTkLabel(master=app,text=path_to_model_test,text_color = "#FCFCFC",text_font=("Bauhaus 93", 12))
test_sample_label = ct.CTkLabel(master=app,text=path_to_sample_test,text_color = "#FCFCFC",text_font=("Bauhaus 93", 12))

button_test_start = ct.CTkButton(master=app,width=250,height=32*2,border_width=0,corner_radius=5,text="Start",fg_color="#FF5F5F",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 16),command=test_start)

#Train

button_main_meun_train = ct.CTkButton(master=app,width=120,height=32*2,border_width=0,corner_radius=5,text="Main Menu",fg_color="#FCFCFC",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 20),command=main_menu_train)

train_model_button = ct.CTkButton(master=app,width=250,height=32*2,border_width=0,corner_radius=5,text="Choose the model",fg_color="#FF5F5F",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 16),command=Path_model_train)
train_bening_sample_button = ct.CTkButton(master=app,width=250,height=32*2,border_width=0,corner_radius=5,text="Bening sample",fg_color="#FF5F5F",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 16),command=Path_sample_bening_train)
train_malicious_sample_button = ct.CTkButton(master=app,width=250,height=32*2,border_width=0,corner_radius=5,text="Malicious sample",fg_color="#FF5F5F",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 16),command=Path_sample_malicious_train)

train_bening_sample_label = ct.CTkLabel(master=app,text=path_to_sample_bening_train,text_color = "#FCFCFC",text_font=("Bauhaus 93", 12))
train_malicious_sample_label = ct.CTkLabel(master=app,text=path_to_sample_malicious_train,text_color = "#FCFCFC",text_font=("Bauhaus 93", 12))
train_model_label = ct.CTkLabel(master=app,text=path_to_model_train,text_color = "#FCFCFC",text_font=("Bauhaus 93", 12))

radiobutton_1 = ct.CTkRadioButton(master=app, text="New model",command=radiobutton_event, text_font=("Bauhaus 93", 12), variable= rad_train, value=1, text_color = "#FCFCFC", fg_color = "#FF5F5F", hover_color='#83FFE6')
radiobutton_2 = ct.CTkRadioButton(master=app, text="Old model",command=radiobutton_event, text_font=("Bauhaus 93", 12), variable= rad_train, value=2, text_color = "#FCFCFC", fg_color = "#FF5F5F", hover_color='#83FFE6')

save_train = ct.CTkCheckBox(master=app, text="Save model",  variable=chel_train, onvalue=1, offvalue=0, text_font=("Bauhaus 93", 12),command=show_train,text_color = "#FCFCFC", fg_color = "#FF5F5F",hover_color ='#83FFE6')

save_entry = ct.CTkEntry(master=app,placeholder_text="Name of model",width=140,height=25,border_width=2,corner_radius=10,text_font=("Bauhaus 93", 12),text_color = "#FCFCFC", fg_color = "#2C2C2C")

check_model_accuracy = ct.CTkCheckBox(master=app, text="Plot accuracy model",  variable=check_mod_acc, onvalue=1, offvalue=0, text_font=("Bauhaus 93", 12),text_color = "#FCFCFC", fg_color = "#FF5F5F",hover_color ='#83FFE6')
check_model_loss = ct.CTkCheckBox(master=app, text="Plot loss function model",  variable=check_mod_loss, onvalue=1, offvalue=0, text_font=("Bauhaus 93", 12),text_color = "#FCFCFC", fg_color = "#FF5F5F",hover_color ='#83FFE6')
check_confusion_matrix = ct.CTkCheckBox(master=app, text="Plot confusion matrix",  variable=check_conf_mat, onvalue=1, offvalue=0, text_font=("Bauhaus 93", 12),text_color = "#FCFCFC", fg_color = "#FF5F5F",hover_color ='#83FFE6')

epochs_entry = ct.CTkEntry(master=app,placeholder_text="Amount of epochs",width=160,height=25,border_width=2,corner_radius=10,text_font=("Bauhaus 93", 12),text_color = "#FCFCFC", fg_color = "#2C2C2C")

train_button_start= ct.CTkButton(master=app,width=120,height=32*2,border_width=0,corner_radius=5,text="Start",fg_color="#FF5F5F",text_color = "#2C2C2C",hover_color='#83FFE6',text_font=("Bauhaus 93", 20),command=train_start)

Test = Text(app, height=20, width=61, font=("Verdana", 14))

Train = Text(app, height=17, width=40, font=("Verdana", 14))

button_train.place(x=50, y=20)
button_test.place(x=200, y=20)
button_exit.place(x =50, y=100)

app.mainloop()
