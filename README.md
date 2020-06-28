# Project Video : https://www.youtube.com/watch?v=9fOigJRpP88
# Motivation: 
This project is made for the third year second semester System Development(CSE-3200) course.
       
   ![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/8.png)
          
   ![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/9.png)
   
# 1	Introduction
Flow chart generator An AI (image processing) based system can convert hand drawn flow chart into real flowchart which can be edited using Microsoft Word. Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM off-line HTR dataset. This Neural Network (NN) model recognizes the text contained in the images of segmented words as shown in the illustration below. As these word-images are smaller than images of complete text-lines, the NN can be kept small and training on the CPU is feasible. A user-friendly graphical interface is associated with the system to make it simple for the user to operate the system
            
# 2 Objectives

The goal of the project is to detect hand-drawn flow-chart and handwriting and convert it into an editable format. Mainly examiner can turn their hand-written question script into an optical character format.
            
# 3	Related Work

Early methods consisted in recognizing isolated characters, or in over-segmenting the image, and scoring groups of segments as characters (e.g. Bengio et al., 1995, Knerr et al., 1999). They were progressively replaced by the sliding window approach, in which features are extracted from vertical frames of the line image (Kaltenmeier, 1993). This approach formulates the problem as a sequence to sequence transduction. The two-dimensional nature of the image may be encoded with convolutional neural networks (Bluche et al., 2013) or by defining relevant features (e.g. Bianne et al. 2011).
            
# 4  Design (Data Flow Diagram)
            
   ![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/1.png)
            
            Figure 1: Flow diagram of Hand-drawn handwriting and flowchart detection system

The System consists of three main components -

•	Handwriting  detection model 
•	Flow-chart  detection model 
•	A graphical user interface (GUI)


First, the system chooses if the file is of flow-chart or handwriting. Then if it is handwriting it runs the NN model to detect the words at each line. Then if it is flow-chart then it detects all elements of flow-chart. Finally, it puts the result into the user-chosen formats.
Again, the Graphical User Interface (GUI) makes the system interactive for a user to use. Users can choose the format if it is an image or text or doc file of Microsoft word. 
 
# 4	Methodology

We will build a Neural Network (NN) which is trained on word-images from the IAM dataset. As the input layer (and therefore also all the other layers) can be kept small for word-images, NN-training is feasible on the CPU (of course, a GPU would be better). This implementation is the bare minimum that is needed for HTR using TF. 
 
![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/2.png)

        	 Fig. 2: Image of word (taken from IAM) and its transcription into digital text.
We use a NN for our task. It consists of convolutional NN (CNN) 

layers, recurrent NN (RNN) layers and a final Connectionist Temporal Classification (CTC) layer. Fig. 3  shows an overview of our HTR system.We use a NN for our task. It consists of convolutional NN (CNN) layers, recurrent NN (RNN) layers and a final Connectionist Temporal Classification (CTC) layer. Fig. 3  shows an overview of our HTR system.

         
         	
 
 # 5.1 Model Overview

 
![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/3.png)

            Fig. 3: Overview of the NN operations (green) and the data flow through the NN (pink).
            
We can also view the NN in a more formal way as a function (see Eq. 1) which maps an   image (or matrix) M of size W×H to a character sequence (c1, c2, …) with a length between 0 and L. As you can see, the text is recognized on character-level, therefore words or texts not contained in the training data can be recognized too (as long as the individual characters get correctly classified).

 
 ![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/4.png)
 
            Eq. 1: The NN written as a mathematical function which maps an image M to a character sequence (c1, c2, …).

# 5.2 Data
Input: it is a gray-value image of size 128×32. Usually, the images from the dataset do not have exactly this size, therefore we resize it (without distortion) until it either has a width of 128 or a height of 32. Then, we copy the image into a (white) target image of size 128×32. This process is shown in Fig. 4. Finally, we normalize the gray-values of the image which simplifies the task for the NN. Data augmentation can easily be integrated by copying the image to random positions instead of aligning it to the left or by randomly resizing the image.

  ![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/5.png)

            Fig. 4: Left: an image from the dataset with an arbitrary size. It is scaled to fit the target image of size 128×32, the empty part of the target image is filled with white color.


 ![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/6.png)

            Fig. 5: Top: 256 feature per time-step are computed by the CNN layers. 
Middle: input image. Bottom: plot of the 32nd feature, which has a high correlation with the occurrence of the character “e” in the image.
CNN output: Fig. 5 shows the output of the CNN layers which is a sequence of length 32. Each entry contains 256 features. Of course, these features are further processed by the RNN layers, however, some features already show a high correlation with certain high-level properties of the input image: there are features which have a high correlation with characters (e.g. “e”), or with duplicate characters (e.g. “tt”), or with character-properties such as loops (as contained in handwritten “l”s or “e”s).

  #          5.3 Flow-chart detection
To detect the flow-chart we used open-cv function. Our idea is very simple.  First we choose each element by detecting the contours.                                                                                                                     
At first our system will initialize the shape name and approximate the contour.
After that, it computes the bounding box of the contour and use the
bounding box to compute the aspect ratio.
 	  	If the shape is a triangle, it will have 3 vertices
      	 	Else if the shape has 4 vertices, it is either a square or
         	  	a rectangle
 
Again, checking  normal square or rectangle.       
     	           If a square will have an aspect ratio that is approximately.
     	           Else if equal to one, otherwise, the shape is a rectangle.
       	     	Else if the shape is a pentagon, it will have 5 vertices
   	           Else we assume the shape is a circle or ellipse

 Finally  return the name of the shape
Using such measures we can simply detect the element of the flow-chart.

  #          5.4 Dataset
[IAM Dataset] (http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database)       
* [Model1 - word_model.png] Train on WORD unit of dataset.
* [Model2 - line_model.png] Train on LINE unit of dataset.
# 6	Implementation

   #         6.1	Graphical User Interface (GUI)

The user interface has all the options needed for the administration and other debugging purpose so that, we do not need to edit code for any management.
 

![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/7.png)

            Figure 6: Overall user interface view






Primarily when someone will open out project then can either choose to convert hand-drawn flow chart or handwriting.  If someone chooses flow-chart then it will convert the flow-chart in the given format : 


 

![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/8.png)

            Fig 7: conversion of flow-chart from hand-drawn image with our system





When someone choose handwriting option then this system will convert it like below:


 

![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/9.png)

            Fig 8: conversion of text from hand-drawn image with our system


When system task will continue then progress-bar will appear and after that finishing the task a simple notification will be given that the task is finished.

 ![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/10.png)

            Fig 9: progress-bar
            
  ![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/11.png)
 
     Fig 10: Complete of conversion 

The overall use case of the total software will be:


![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/12.png)
 
            Fig 11: conversion of text from hand-drawn image with our system



If user wants he/she can use Microsoft word to manually check the errors that system makes. 

Our system can directly send the generated text to word file and save accordingly.	


 
![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/13.png)

            Fig 12: Use case of what format to choose when to save a file in our system


 
![alt text](https://github.com/Dream-kid/Flow-Chart-Generator-From-Hand-Drawn-Chart-And-Handwriting-Recognition/blob/master/project%20images/14.png)

            Fig 13: Use case of flow-chart into docx file into our system
 


Libraries used for graphical user interface:

1.	TkInter


  #  6.2 Result
Test on IAM dataset:

|  Model 	 | Test Unit 	| Number of samples         | CER(%)	 | WER(%)     | 
| :-                   | :-                 |     :---:                                 |  ---:                |  ---: 		|
|  WORD         | WORD        | 19289                                 | 10.39             |   26.97        | 
|  WORD   	| LINE            | 2192          		           | 21.73             | 46.00          | 
|  LINE   	| LINE            | 2192                                   | 08.32             | 28.99          | 

            6.3 Train
[Google colab]

  #  7	Conclusion and Recommendation

The designed algorithm was effectively able to detect the type of flow-chart elements. If the system gets optical character it’s output becomes 100% accurate. But people's handwriting is different from each other. To increase the accuracy 3 models are used to predict the handwriting and the best one is taken as output also It uses the backtrack. So this process is relatively slow but the accuracy is higher. Future generations can increase their speed by using optimal models or other techniques. In the case of flow chart another complex symbol can be added they can use the open-cv library or other techniques. Also, future generations can detect handwriting and flowchart by using low-intensity picture samples.


 #    8 References
[1]  Handwritten text recognition in historical documents / von Harald Scheidl Author Scheidl, Harald Thesis advisor Sablatnig, Robert Published Wien, 

# Author
Soarov Chakraborty, Shourov Paul    
soarovchakraborty@gmail.com     
Student at Department of Computer Science and Engineering                                                 
Khulna University of Engineering & Technology, Khulna                             
Bangladesh


Supervised under                                         
Dr. Sk. Mohammad Masudul Ahsan                                                                                         
Professor                             
Dept. of Computer Science and Engineering          
Khulna University of Engineering & Technology              




