import os
import sys
import glob
import numpy as np
from aletheialib.models import NN 

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

def stegano_predict(image_list, model_path):
    pred_list = list()

    nn = load_stegano_model(model_path)
    # predicting images
    try:
        pred = nn.predict(image_list, batch=32)
        #print("steganography model :" + model_path)
        for i in range(len(pred)):
            #print("steganography prediction percent : " + str(round(100*pred[i], 3)))
            pred_list.append(pred[i])
"""
            if pred[i]>=0.5:
                #print(image_list[i] + " -\033[31m Steganography\033[0m")
                pred_list.append(pred[i])
            else:
                #print(image_list[i] + " -\033[31m Non-Steganography\033[0m")
                pred_list.append(pred[i ])
"""
        return pred_list

    except Exception as e:
        print('fail to predict' + str(e))
        return 0


def load_stegano_model(model_path):
    try:
        nn = NN("effnetb0")
        nn.load_model(model_path)
    except Exception as e:
        print("Error :", e)

    return nn


def get_binary_matrix(y_val, y_pred):
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print('accuracy :', accuracy)
    print('recall :', recall)
    print('precision :', precision)
    print('f1-score :', f1)

    return accuracy, recall, precision, f1


def get_multiclass_matrix(y_val, y_pred):
    print("<< confusion matrix >>")
    print("="*30)
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    sns.heatmap(cm, annot=True, cmap='Blues')


    print("\n\n<< classification report >>")
    print("="*60)
    print(classification_report(y_val, y_pred))

    acc = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred, average='macro')
    precision = precision_score(y_val, y_pred, average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')

    return acc, recall, precision, f1


if __name__ == '__main__':
    model_list = ['effnetb0-hill.h5', 'effnetb0-hilluniw.h5', 'effnetb0-juniw.h5', 'effnetb0-lsbm.h5', 'effnetb0-lsbr.h5', 'effnetb0-nsf5.h5', 'effnetb0-outguess.h5', 'effnetb0-steganogan.h5', 'effnetb0-steghide.h5', 'effnetb0-uniw.h5']    

    model_root_path = '/lockard_ai/conf/model/aletheia-models/'

    image_dir = '/lockard_ai/works/Steganography/aletheia-steganalysis/sample_images/alaska2jpg/'
    
    if os.path.isdir(image_dir):
        image_list = glob.glob(os.path.join(image_dir, '*.*'))
    else: 
        print("Error : image path wrong.")

    for m in model_list:
        print("# model : ", m)
        model_path = os.path.join(model_root_path, m)
        pred_list = stegano_predict(image_list, model_path)

    
