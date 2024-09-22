from tensorflow.keras.optimizers import Adamax, Adam
import tensorflow as tf
import prepare_data
import glob
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from vit_keras import vit, utils
from vit_datagenerator import CustomDataGenerator
import time
from scipy.stats import mode

WINDOW_SIZE = 29

def apply_mode_filter(data):
    half_window = int((WINDOW_SIZE-1)/2)
    
    for i in range(len(data)):
        if i > 0 and data[i] != data[i-1]:
            start = max(0, i-half_window)
            end = min(len(data), i+half_window+1)

            window_data = data[start:end]
            mode_value = mode(window_data)[0]
            data[i] = mode_value

    return data

#loaded_model = tf.keras.models.load_model('./gru_num14_yeschange/my_grumodel_num14_epo253_yesrandomchange')
loaded_model = tf.keras.models.load_model('./lstm_num14_yeschange/my_lstmmodel_num14_epo211_yesrandomchange')
#loaded_model = tf.keras.models.load_model('./lstm_num14_nochange/my_lstmmodel_num14_epo160_nochange')
#loaded_model = tf.keras.models.load_model('./train_model/model')
#loaded_model.load_weights('./train_model/best_weights')

#my
#loaded_model.compile(loss='binary_crossentropy', optimizer = Adam(learning_rate=0.001), metrics=['accuracy'])
loaded_model.compile(loss='mean_squared_error', optimizer = Adam(learning_rate=0.001), metrics=['accuracy'])
#sekiya
#loaded_model.compile(loss='mean_squared_error', optimizer = Adamax(), metrics=['accuracy'])

#testdata
DATA_DIR = './imagefile'
videos_test = sorted(glob.glob(DATA_DIR+'/taimatu/*')+glob.glob(DATA_DIR+'/sukegawa/*'))
#videos_test = sorted(glob.glob(DATA_DIR+'/matukane/*')+glob.glob(DATA_DIR+'/sukegawa/*'))
#videos_test = sorted(glob.glob(DATA_DIR+'/sako/record_on_1froom_20231113_1330')+glob.glob(DATA_DIR+'/sako/record_on_lab_20231113_1330')+glob.glob(DATA_DIR+'/sekiya/bathroom_1')+glob.glob(DATA_DIR+'/sekiya/myroom_5'))
#print(videos_test)
#videos_test = sorted(glob.glob('./record_off_myroom_20231114_2100'))
#videos_test = sorted(glob.glob('./record_off_myroom_20231116_2200'))
#videos_test = sorted(glob.glob('./record_on_living_20231114_2200'))
#videos_test = sorted(glob.glob('./record_on_living_20231213_1000'))
#videos_test = sorted(glob.glob('./record_on_lab_20240429_1400_5050'))
#videos_test = sorted(glob.glob(DATA_DIR+'/sekiya/bathroom_1'))
inputs_test_path, labels_test_path = prepare_data.videotopath(videos_test)
#print(inputs_test_path)
#gazouread
inputs_test = prepare_data.imageread2(inputs_test_path)
#brightchange
#change_inputs_test = prepare_data.changebright(inputs_test)
#inputs_test = inputs_test+change_inputs_test
#inputs_test = change_inputs_test
print(len(inputs_test))

"""changes = []
for input_test in inputs_test:
    change = cv2.convertScaleAbs(input_test*255, alpha=1.5, beta=0)
    change = np.asarray(change).astype('float32') / 255
    changes.append(change)
inputs_test = changes"""


#padhingu
sequence_X_test = prepare_data.padhing(inputs_test)
#correctlabel
Y_test = prepare_data.correctlabel(labels_test_path)
#change_Y_test = prepare_data.addcorrectlabel(labels_test_path)
#Y_test = Y_test+change_Y_test
#Y_test = change_Y_test
print(len(Y_test))
#padhingu of correctlabel
sequence_Y_test = prepare_data.padhing(Y_test)




#test
score = loaded_model.evaluate(sequence_X_test, sequence_Y_test, batch_size=4)
#score2 = loaded_model2.evaluate(sequence_X_test, sequence_Y_test, batch_size=4)
#score = loaded_model.evaluate(test_generator, batch_size=1)

print('loss: ', score[0])
print('accuracy: ', score[1])

#print('loss2: ', score2[0])
#print('accuracy2: ', score2[1])


#predict f1score



#pred.shape=(48, 345, 1)
pred = loaded_model(sequence_X_test)
#chpred = pred

#pred2 = loaded_model2.predict(sequence_X_test)
#chpred2 = pred2
#chpred2 = pred


#print(pred)

pred_touch = np.where(pred > 0.7, 1, 0)
#chpred_touch = np.where(chpred > 0.5, 1, 0)
#chpred_touch2 = np.where(chpred2 > 0.5, 1, 0)

start = time.time()

chpred_touch = []

for data in pred_touch:
    chpred_touch.append(apply_mode_filter(data))

chpred_touch = np.array(chpred_touch) 


#1frame
"""for data in chpred_touch:
    for i in range(1, chpred_touch.shape[1]-1):
        if ((data[i-1]!=data[i]) and (data[i]!=data[i+1])):
            #print(i)
            data[i] = data[i-1]"""

"""for data in chpred_touch2:
    for i in range(1, chpred_touch2.shape[1]-1):
        if ((data[i-1]!=data[i]) and (data[i]!=data[i+1])):
            #print(i)
            data[i] = data[i-1]"""

#2frame
"""for data in chpred_touch:
    for i in range(1, chpred_touch.shape[1]-2):
        if ((data[i-1]!=data[i]) and (data[i]!=data[i+1])):
            #print(i)
            data[i] = data[i-1]
        elif ((data[i-1]!=data[i]) and (data[i]!=data[i+2])):
            data[i] = data[i-1]"""
#3frame
"""for data in chpred_touch:
    for i in range(1, chpred_touch.shape[1]-3):
        if (data[i-1]!=data[i]):
            count=0
            if (data[i]!=data[i+1]):
                count = count + 1
            if ( data[i] != data[i+2]):
                count = count + 1
            if ( data[i] != data[i+3]):
                count = count + 1
            if (count > 1):
                data[i] = data[i-1]"""

#5frame
"""for data in chpred_touch:
    for i in range(1, chpred_touch.shape[1]-5):
        if (data[i-1]!=data[i]):
            count=0
            if (data[i]!=data[i+1]):
                count = count + 1
            if ( data[i] != data[i+2]):
                count = count + 1
            if ( data[i] != data[i+3]):
                count = count + 1
            if ( data[i] != data[i+4]):
                count = count + 1
            if ( data[i] != data[i+5]):
                count = count + 1
            if (count > 2):
                data[i] = data[i-1]"""

#7frame
"""for data in chpred_touch:
    for i in range(1, chpred_touch.shape[1]-5):
        if (data[i-1]!=data[i]):
            count=0
            if (data[i]!=data[i+1]):
                count = count + 1
            if ( data[i] != data[i+2]):
                count = count + 1
            if ( data[i] != data[i+3]):
                count = count + 1
            if ( data[i] != data[i+4]):
                count = count + 1
            if ( data[i] != data[i+5]):
                count = count + 1
            if ( data[i] != data[i+6]):
                count = count + 1
            if ( data[i] != data[i+7]):
                count = count + 1
            if (count > 3):
                data[i] = data[i-1]"""

#9frame
"""for data in chpred_touch:
    for i in range(1, chpred_touch.shape[1]-5):
        if (data[i-1]!=data[i]):
            count=0
            if (data[i]!=data[i+1]):
                count = count + 1
            if ( data[i] != data[i+2]):
                count = count + 1
            if ( data[i] != data[i+3]):
                count = count + 1
            if ( data[i] != data[i+4]):
                count = count + 1
            if ( data[i] != data[i+5]):
                count = count + 1
            if ( data[i] != data[i+6]):
                count = count + 1
            if ( data[i] != data[i+7]):
                count = count + 1
            if ( data[i] != data[i+8]):
                count = count + 1
            if ( data[i] != data[i+9]):
                count = count + 1
            if (count > 4):
                data[i] = data[i-1]"""

#11frame
"""for data in chpred_touch:
    for i in range(1, chpred_touch.shape[1]-5):
        if (data[i-1]!=data[i]):
            count=0
            if (data[i]!=data[i+1]):
                count = count + 1
            if ( data[i] != data[i+2]):
                count = count + 1
            if ( data[i] != data[i+3]):
                count = count + 1
            if ( data[i] != data[i+4]):
                count = count + 1
            if ( data[i] != data[i+5]):
                count = count + 1
            if ( data[i] != data[i+6]):
                count = count + 1
            if ( data[i] != data[i+7]):
                count = count + 1
            if ( data[i] != data[i+8]):
                count = count + 1
            if ( data[i] != data[i+9]):
                count = count + 1
            if ( data[i] != data[i+10]):
                count = count + 1
            if ( data[i] != data[i+11]):
                count = count + 1
            if (count > 5):
                data[i] = data[i-1]"""

#13frame
"""for data in chpred_touch:
    for i in range(1, chpred_touch.shape[1]-5):
        if (data[i-1]!=data[i]):
            count=0
            if (data[i]!=data[i+1]):
                count = count + 1
            if ( data[i] != data[i+2]):
                count = count + 1
            if ( data[i] != data[i+3]):
                count = count + 1
            if ( data[i] != data[i+4]):
                count = count + 1
            if ( data[i] != data[i+5]):
                count = count + 1
            if ( data[i] != data[i+6]):
                count = count + 1
            if ( data[i] != data[i+7]):
                count = count + 1
            if ( data[i] != data[i+8]):
                count = count + 1
            if ( data[i] != data[i+9]):
                count = count + 1
            if ( data[i] != data[i+10]):
                count = count + 1
            if ( data[i] != data[i+11]):
                count = count + 1
            if ( data[i] != data[i+12]):
                count = count + 1
            if ( data[i] != data[i+13]):
                count = count + 1
            if (count > 6):
                data[i] = data[i-1]"""

#15frame
"""for data in chpred_touch:
    for i in range(1, chpred_touch.shape[1]-5):
        if (data[i-1]!=data[i]):
            count=0
            if (data[i]!=data[i+1]):
                count = count + 1
            if ( data[i] != data[i+2]):
                count = count + 1
            if ( data[i] != data[i+3]):
                count = count + 1
            if ( data[i] != data[i+4]):
                count = count + 1
            if ( data[i] != data[i+5]):
                count = count + 1
            if ( data[i] != data[i+6]):
                count = count + 1
            if ( data[i] != data[i+7]):
                count = count + 1
            if ( data[i] != data[i+8]):
                count = count + 1
            if ( data[i] != data[i+9]):
                count = count + 1
            if ( data[i] != data[i+10]):
                count = count + 1
            if ( data[i] != data[i+11]):
                count = count + 1
            if ( data[i] != data[i+12]):
                count = count + 1
            if ( data[i] != data[i+13]):
                count = count + 1
            if ( data[i] != data[i+14]):
                count = count + 1
            if ( data[i] != data[i+15]):
                count = count + 1
            if (count > 7):
                data[i] = data[i-1]"""


end = time.time()

print('time: ', end-start)

reshape_pred = pred_touch.reshape((sequence_X_test.shape[0], -1))
reshape_chpred = chpred_touch.reshape((sequence_X_test.shape[0], -1))
#reshape_chpred2 = chpred_touch2.reshape((sequence_X_test.shape[0], -1))

reshape_label = sequence_Y_test.reshape((sequence_X_test.shape[0], -1))
label_touch = np.where(reshape_label > 0.5, 1, 0)

join_label = []
join_predict = []
join_chpredict = []
#join_chpredict2 = []
for i in label_touch:
    join_label.extend(i)
for j in reshape_pred:
    join_predict.extend(j)
for k in reshape_chpred:
    join_chpredict.extend(k)
"""for p in reshape_chpred2:
    join_chpredict2.extend(p)"""
#print(join_label[:sequence_X_test.shape[1]])
#print(join_predict[:sequence_X_test.shape[1]])
#print(join_chpredict[:sequence_X_test.shape[1]])
for i in range(sequence_X_test.shape[0]):
    print(join_label[i*sequence_X_test.shape[1]:(i+1)*sequence_X_test.shape[1]])
    #print(join_predict[i*sequence_X_test.shape[1]:(i+1)*sequence_X_test.shape[1]])
    print(join_chpredict[i*sequence_X_test.shape[1]:(i+1)*sequence_X_test.shape[1]])
    #print(join_chpredict2[i*sequence_X_test.shape[1]:(i+1)*sequence_X_test.shape[1]])
    print(i)

count_one = 0
touch = 'N'
for i, chp in enumerate(reshape_chpred):
    count_one = 0
    touch = 'N'
    for j in range(1, len(chp)):
        if ((chp[j]!=chp[j-1]) and touch == 'N'):
            touch = 'Y'
            count_one = count_one+1
        elif (touch == 'Y' and chp[j]==1):
            count_one = count_one+1
        elif (touch == 'Y' and chp[j]==0):
            print('i, count_one', i, count_one)
            count_one = 0
            touch = 'N'
        if ( j==(len(chp)-1) and count_one>0):
            print('i, count_one', i, count_one)
            count_one = 0
            touch = 'N'

#print(join_label)
#print(join_predict)
"""for i, data in enumerate(join_predict[:sequence_X_test.shape[1]]):
    if data == 1:
        print(i)"""

"""for i in range(reshape_chpred.shape[0]):
    for j in range(reshape_chpred.shape[1]):
        if (reshape_chpred[i][j]!=reshape_chpred2[i][j]):
            print(i, j)"""

join_cm = confusion_matrix(join_label, join_predict)
print(join_cm)
join_f1 = f1_score(join_label, join_predict)
print("f1:", join_f1)
join_precision = precision_score(join_label, join_predict)
print("precision: ", join_precision)
join_recall = recall_score(join_label, join_predict)
print("recall: ", join_recall)

chjoin_cm = confusion_matrix(join_label, join_chpredict)
print(chjoin_cm)
chjoin_f1 = f1_score(join_label, join_chpredict)
print("ch_f1:", chjoin_f1)
chjoin_precision = precision_score(join_label, join_chpredict)
print("ch_precision: ", chjoin_precision)
chjoin_recall = recall_score(join_label, join_chpredict)
print("ch_recall: ", chjoin_recall)


