import numpy as np
from Utils import *
from WordSegmentation import wordSegmentation, prepareImg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from Preprocessor import preprocess
from keras.models import model_from_json
import shutil
from keras import backend as K
from keras.utils import plot_model
from Spell import correction_list


def how(gd):
	#l_model, l_model_predict = line_model()
	#with open('line_model_predict.json', 'w') as f:
	#	f.write(l_model_predict.to_json())
    with open('Resource/line_model_predict.json', 'r') as f:
        l_model_predict = model_from_json(f.read())
    with open('Resource/word_model_predict.json', 'r') as f:
        w_model_predict = model_from_json(f.read())
	#plot_model(l_model_predict, to_file='line_model.png', show_shapes=True, show_layer_names=True)
    w_model_predict.load_weights('Resource/iam_words--15--1.791.h5')
    l_model_predict.load_weights('Resource/iam_lines--12--17.373.h5')
    test_img = gd#############################
	
    img = prepareImg(cv2.imread(test_img), 64)
    img2 = img.copy()
    res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        (x, y, w, h) = wordBox
        cv2.imwrite('tmp/%d.png'%j, wordImg)
        cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),1) # draw bounding box in summary image

    cv2.imwrite('Resource/summary.png', img2)
    plt.imshow(img2)
    imgFiles = os.listdir('tmp')
    imgFiles = sorted(imgFiles)
    pred_line = []
    for f in imgFiles:
        pred_line.append(predict_image(w_model_predict, 'tmp/'+f, True))
    print('-----------PREDICT-------------') 
    text1 =' '.join(pred_line)
    print('[Word model]: '+' ' + text1)

    pred_line = correction_list(pred_line)
    text2=' '.join(pred_line)
    print('[Word model with spell]: '+ text2)

    text3=	predict_image(l_model_predict, test_img, False)
    print('[Line model]: ' + text3)
    
    plt.show()
    shutil.rmtree('tmp')
    """
    f1= open("output/output1.txt","a")
    f1.write(text1)
    f1.write("\n")
    f1.close()
   
    f1= open("output/output2.txt","a")
    f1.write(text2)
    f1.write("\n")
    f1.close()
    """
    f1= open("output/output3.txt","a")
    f1.write(text3)
    f1.write("\n")
    f1.close()

    
   
