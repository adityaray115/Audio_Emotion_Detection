from tkinter import *
import pickle
import random
import librosa
import numpy as np
import os

from tensorflow.keras.models import load_model

root = Tk()
root.geometry("300x250")

rel_path=os.path.dirname(os.path.realpath(__file__))

def feature_extraction(file_name):
    audio,sample_rate=librosa.load(file_name,res_type='kaiser_fast')
    mfcc_features=librosa.feature.mfcc(y=audio,sr=sample_rate, n_mfcc=60)
    mfcc_scaled_features=np.mean(mfcc_features.T,axis=0)

    return mfcc_scaled_features

loaded_model_dTree = pickle.load(open(str(rel_path+"\\Trained_model_dTree.h5"), 'rb'))
loaded_model_CNN = load_model(str(rel_path+"\\Trained_model_CNN.h5"))
loaded_model_MLP = pickle.load(open(str(rel_path+"\\Trained_model_MLP.h5"), 'rb'))

def show():
    selected_emotion= str(emotion_selection.get())
    selected_model=str(model_selection.get())
    if selected_emotion == 'Random':selected_emotion=random.choice(emotion_options[1:])
    if selected_model == 'Random':selected_model=random.choice(model_options[1:])
    filename="\\Emotions\\"+selected_emotion
    file_list=os.listdir(rel_path+filename)
    filename=rel_path+filename+"\\"+random.choice(file_list)
    actual_emotion=selected_emotion
    predicted_emotion=''
    actual_tag.config(text = "Actual Emotion:  "+actual_emotion)
    emotion_selection.set(selected_emotion)
    model_selection.set(selected_model)
    prediction_feature=feature_extraction(filename)
    prediction_feature=prediction_feature.reshape(1,-1)
    rev_keys={0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Suprised"}
    if selected_model=='Decision Tree':
        predicted_emotion=rev_keys[np.argmax(loaded_model_dTree.predict(prediction_feature),axis=1)[0]]
    elif selected_model=='CNN':
        prediction_feature=np.expand_dims(prediction_feature, axis=2)
        predicted_emotion=rev_keys[np.argmax(loaded_model_CNN.predict(prediction_feature),axis=1)[0]]
    elif selected_model=='MLP':
        predicted_emotion=rev_keys[np.argmax(loaded_model_MLP.predict(prediction_feature),axis=1)[0]]
    predicted_tag.config(text = "Actual Emotion:  "+predicted_emotion)
    



emotion_frame = Frame(root)
emotion_frame.pack( side = TOP )

emotion_tag = Label( emotion_frame , text = "Select Emotion to predict" ).pack(side=LEFT,pady=20)

# Dropdown menu options
emotion_options = [
    "Random",
    "Angry",
    "Disgusted",
    "Fearful",
    "Happy",
    "Neutral",
    "Sad",
    "Suprised"
]
emotion_selection = StringVar()
emotion_selection.set( "Random" )
drop = OptionMenu( emotion_frame , emotion_selection , *emotion_options ).pack(side=LEFT)

model_frame = Frame(root)
model_frame.pack( side = TOP,pady=12 )

emotion_tag = Label( model_frame , text = "Select Emotion to predict" ).pack(side=LEFT)

# Dropdown menu options
model_options = [
    "Random",
    "Decision Tree",
    "CNN",
    "MLP",
]
model_selection = StringVar()
model_selection.set( "Random" )
drop = OptionMenu( model_frame , model_selection , *model_options ).pack(side=LEFT)

button = Button( root , text = "click Me" , command = show ).pack(side=TOP)
actual_tag = Label( root , text = "Actual Emotion:  " )
actual_tag.pack(side=TOP,pady=12)
predicted_tag = Label( root , text = "Actual Emotion:  " )
predicted_tag.pack(side=TOP)

root.mainloop()