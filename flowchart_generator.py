# import the required packages
import imutils
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytesseract
import tkinter as tk


thickness = 10
#to extract text from image
def extract_text(image,imagename):#variable , name of file
    pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    # load the image and convert it to grayscale
    cv2.imwrite(imagename,image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # write the grayscale image to disk as a temporary file so we can apply OCR to it
    cv2.imwrite("ocr_gray.jpg", gray)
    # load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
    text = pytesseract.image_to_string(Image.open(imagename))
    #print(text)
    return text
#for identifying shapes 
class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)#aspect ratio
        (z,v),(MA,ma),angle = cv2.fitEllipse(c)
        if ar*50 < 50:
            shape = "arrow"
        else:
    
            # if the shape is a triangle, it will have 3 vertices
            if len(approx) == 3:
                shape = "triangle"
    
            # if the shape has 4 vertices, it is either a square or
            # a rectangle
            elif len(approx) == 4:
                if (int(angle) >= 0 and int(angle) <=70 )or ar<=1.5:#Diamond 
                    shape = "Diamond"
                elif int(angle) > 70 and int(angle) <=85:#parallelogram
                    shape = "parallelogram"
                else :#normal square or rect       
                    # a square will have an aspect ratio that is approximately
                    # equal to one, otherwise, the shape is a rectangle
                    shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    
            # if the shape is a pentagon, it will have 5 vertices
            elif len(approx) == 5:
                shape = "pentagon"
    
            # otherwise, we assume the shape is a circle or ellipse
            else:
                #if aspect ratio > 1 then ellipse
                if ar > 1.05 :
                    shape = "ellipse"
                else:
                    a = np.pi *( (w/2)**2)
                    shape = "circle"

        # return the name of the shape
        return shape
#to turn image that was read using openCv (bgr) to (rgb)     
def cv2mat(imagename , imagevar):
    lol='flowchart/output/'
    lol=lol+imagename
    imagename=lol
    cv2.imwrite(imagename,imagevar) 
    bgr_img = cv2.imread(imagename)  #reading image using openCV
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    #plt.imshow(rgb_img)
    return rgb_img

#to show images on console
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(3*fig.get_size_inches()) * n_images)
    plt.show()
    
#turn drawn flowchart to digital version
def flowchart(image,outputname):
    white = 255*np.ones_like(image)
    white2 = image.copy() 
    blurred = cv2.GaussianBlur(image, (5,5), 0)
    #resize the image
    resized = imutils.resize(blurred,width = 2* image.shape[1])
    ratio = image.shape[0] / float(resized.shape[0])
    # convert the resized image to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    #histogram eq
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    equ = clahe.apply(gray)
    equ2 = cv2.fastNlMeansDenoising(equ,None,270,7,21)
    
    # Otsu's thresholding
    blur = cv2.GaussianBlur(equ2,(25,25),0)
    ret3,th3 = cv2.threshold(blur,10,100,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #detect edges
    edge = cv2.Canny(th3,0,255,apertureSize = 3)
    
    #opening image for less noise
    blurred = cv2.GaussianBlur(edge, (9,9), 0)
    kernel = np.ones((3,3),np.uint8)     
    MORPH_OPEN = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts,hierarchy = cv2.findContours(MORPH_OPEN.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()
    
    # loop over the contours
    for i in range(len(cnts)):
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        c=cnts[i]
        if len(cnts[i]) >= 5:
          
            #print('ok')
            area = cv2.contourArea(c)
            (z,q),(MA,ma),angle = cv2.fitEllipse(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            # (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
            (x, y, w, h) = cv2.boundingRect(approx)
            x = int(x*ratio)
            y = int(y*ratio)
            w = int(w*ratio)
            h = int(h*ratio)
            a = int((h/2)*1)
            if ( int(ma) < int(0.5*resized.shape[0]) & int(MA) < int(0.5*resized.shape[1]) )or area > 0.001*(image.shape[0]*image.shape[1]):
                M = cv2.moments(c)
                cX = int((M["m10"] / (M["m00"]) )* ratio)
                cY = int((M["m01"] / (M["m00"])) * ratio)
                shape = sd.detect(c)
    #            print(shape)
            	# multiply the contour (x, y)-coordinates by the resize ratio,
            	# then draw the contours and the name of the  shape on the image
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv2.putText(image, shape, (cX+int(w/2), cY), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 0), 5)
                cropped = white2[y:y+h, x:x+w]           
    #            print(extract_text(cropped,'cropped.jpg'))
                cv2.putText(white, extract_text(cropped,'cropped.jpg'), (cX-int(w/4), cY), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 255), 3)
                if shape == "rectangle":
                    #(x,y) top left corner , (x+w,y+h) bottom right corner
                    draw = cv2.rectangle(white,(x,y),(x+w,y+h),(255,0,0),thickness)
                elif shape == "Diamond":
                    pts = np.array([[cX,cY-int(h/2)],[cX+int(w/2),cY],[cX,cY+int(h/2)],[cX-int(w/2),cY]], np.int32)
                    pts = pts.reshape((-1,1,2))
                    draw = cv2.polylines(white,[pts],True,(255,0,0),thickness)
                elif shape == "parallelogram":
                    pts = np.array([[int(cX-(w/2)+a),int(cY-(h/2))],[int(cX+(w/2)+a),int(cY-(h/2))],[int(cX+(w/2)-a),int(cY+(h/2))],[int(cX-(w/2)-a),int(cY+(h/2))]], np.int32)
                    pts = pts.reshape((-1,1,2))
                    draw = cv2.polylines(white,[pts],True,(255,0,0),thickness)
                elif shape == "ellipse":
                    # center location, (major axis length, minor axis length)
                    #next angle , start angle,end angle ,color,thickness(when -1 fills ellipse)
                    draw = cv2.ellipse(white,(cX,cY),(int(w/2),int(h/2)),0,0,360,255,thickness)
                elif shape == "arrow":
                    draw = cv2.line(white,(x,y+h),(x,y),(255,0,0),5)
                    draw = cv2.line(white,(x,y+h),(x+int(h/6),y+h-int(h/6)),(255,0,0),thickness)
                    draw = cv2.line(white,(x,y+h),(x-int(h/6),y+h-int(h/6)),(255,0,0),thickness)
                else:
                    draw = cv2.rectangle(white,(x,y),(x+w,y+h),(255,0,0),thickness)
        else:
            cv2.drawContours(image,cnts,-1,(0,0,0),-1)
            	 #show the output image
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.namedWindow('white', cv2.WINDOW_NORMAL)
    cv2.imshow("white", white)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    images = [cv2mat(outputname , image) ,cv2mat(outputname , white) ]
    show_images(images, cols = 2, titles = None)

                

def call2(name):
    lol='flowchart/images/'
    lol=lol+name
    #print(lol)
    #lol='input.jpg'
    img = cv2.imread(lol)    
    flowchart(img,'output.jpg')

  
       