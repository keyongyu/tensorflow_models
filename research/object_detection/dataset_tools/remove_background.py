from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
# #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#
# #Load the Image
# imgo = cv2.imread('/home/keyong/Downloads/IMAG2049.jpg')
#
# height, width = imgo.shape[:2]
#
# #Create a mask holder
# mask = np.zeros(imgo.shape[:2],np.uint8)
#
# #Grab Cut the object
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
#
# #Hard Coding the Rect, The object must lie within this rect.
# rect = (10,10,width-30,height-30)
# cv2.grabCut(imgo,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
# mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img1 = imgo*mask[:,:,np.newaxis]
#
# #Get the background
# background = imgo - img1
#
# #Change all pixels in the background that are not black to white
# background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]
#
# #Add the background and the image
# final = background + img1
#
# imgplot = plt.imshow(final)
# imglot.show()
# #cv2.imshow('image', final )
#
# #k = cv2.waitKey(0)
#
# #if k==27:
# #    cv2.destroyAllWindows()

# import pgmagick as pg
#
# def trans_mask_sobel(img):
#     """ Generate a transparency mask for a given image """
#
#     image = pg.Image(img)
#
#     # Find object
#     image.negate()
#     image.write('negate.png')
#     image.edge()
#     image.write('edge.png')
#     image.blur(1)
#     image.write('blur.png')
#     image.threshold(48)
#     image.write('threshold.png')
#     image.adaptiveThreshold(5, 5, 5)
#     image.write('adaptiveThreshold.png')
#
#     # Fill background
#     image.fillColor('magenta')
#     w, h = image.size().width(), image.size().height()
#     image.floodFillColor('0x0', 'magenta')
#     image.floodFillColor('0x0+%s+0' % (w-1), 'magenta')
#     image.floodFillColor('0x0+0+%s' % (h-1), 'magenta')
#     image.floodFillColor('0x0+%s+%s' % (w-1, h-1), 'magenta')
#
#     image.transparent('magenta')
#
#     image.write('mask.png')
#     return image
#
# def alpha_composite(image, mask):
#     """ Composite two images together by overriding one opacity channel """
#
#     compos = pg.Image(mask)
#     compos.composite(
#         image,
#         image.size(),
#         pg.CompositeOperator.CopyOpacityCompositeOp
#     )
#     return compos
#
# def remove_background(filename):
#     """ Remove the background of the image in 'filename' """
#
#     img = pg.Image(filename)
#     transmask = trans_mask_sobel(img)
#     img = alpha_composite(transmask,img)
#     img.trim()
#     img.write('out.png')
#
#
#
# remove_background('/home/keyong/Downloads/IMAG2049.jpg')
# #remove_background('/home/keyong/Downloads/other_examples.jpg')


#import cv2
import numpy as np
#from matplotlib import pyplot as plt

def test():

    def my_show_gray(img,title,idx):
        # plt.subplot(2,3,idx)
        # plt.imshow(img,cmap = 'gray')
        # plt.title(title)
        #print("idx="+str(idx))
        idx+=1


    #== Parameters =======================================================================
    BLUR = 11
    CANNY_THRESH_1 = 0
    CANNY_THRESH_2 = 50
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0,0.0,1.0) # In BGR format

    #== Processing =======================================================================

    #-- Read image -----------------------------------------------------------------------
    img = cv2.imread('/home/keyong/Downloads/IMAG2049.jpg')
    img = img[300:3800, 800:2300]
    #img = cv2.imread('/home/keyong/Downloads/SYxmp.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    my_show_gray(gray,"original",1)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    my_show_gray(gray,"GaussianBlur",2)

    #-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    #edges= cv2.Laplacian(gray,cv2.CV_8UC1)
    edges = cv2.Sobel(gray,cv2.CV_8UC1, 0,1,ksize=5)
    my_show_gray(edges,"laplace",3)
    #cv2.imwrite('/home/keyong/test_edges_0.png', edges)
    edges = cv2.dilate(edges, None)
    my_show_gray(edges,"dilate",4)
    edges = cv2.erode(edges, None)
    my_show_gray(edges,"erode",5)
    #plt.show()
    #cv2.imwrite('/home/keyong/test_edges.png', edges)

    ret, thresh = cv2.threshold(gray, 64, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    plt.imshow(img)
    plt.show()

    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    #cv2.fillConvexPoly(mask, max_contour[0], (255))
    for contour in contour_info:
        print("area = %f"%(contour[2]))
        if contour[1] or contour[2] < 1000:
            continue;
        cv2.fillConvexPoly(mask, contour[0], (255))



    #-- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

    #-- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices,
    img         = img.astype('float32') / 255.0                 #  for easy blending

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit

    plt.imsave('/home/keyong/test_blue.png', masked)
    # split image into channels
    c_red, c_green, c_blue = cv2.split(img)

    # merge with mask got on one of a previous steps
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

    # show on screen (optional in jupiter)
    #%matplotlib inline
    #plt.imshow(img_a)
    #plt.show()

    # save to disk
    cv2.imwrite('/home/keyong/test_1.png', img_a*255)

    # or the same using plt
    plt.imsave('/home/keyong/test_2.png', img_a)

    #cv2.imshow('img', masked)                                   # Displays red, saves blue

    #cv2.waitKey()

#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.
This sample shows interactive image segmentation using grabcut algorithm.
USAGE:
    python grabcut.py <filename>
README FIRST:
    Two windows will show up, one for input and one for output.
    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.
Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground
Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''

# Python 2/3 compatibility

import numpy as np
import cv2 as cv
import sys

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness

def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    # Draw Rectangle
    if event == cv.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        rect_or_mask = 0
        print(" Now press the key 'n' a few times until no further change \n")

    # draw touchup curves

    if event == cv.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("first draw rectangle \n")
        else:
            drawing = True
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)

if __name__ == '__main__':

    # print documentation
    print(__doc__)

    # Loading images
    if len(sys.argv) == 2:
        filename = sys.argv[1] # for drawing purposes
    else:
        print("No input image given, so loading default image, ../data/lena.jpg \n")
        print("Correct Usage: python grabcut.py <filename> \n")
        filename = '/home/keyong/Downloads/IMAG2049.jpg'

    img = cv.imread(filename)
    img2 = img.copy()                               # a copy of original image
    mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
    output = np.zeros(img.shape,np.uint8)           # output image to be shown

    # input and output windows
    cv.namedWindow('output')
    cv.namedWindow('input')
    cv.setMouseCallback('input',onmouse)
    cv.moveWindow('input',img.shape[1]+10,90)

    print(" Instructions: \n")
    print(" Draw a rectangle around the object using right mouse button \n")

    while(1):

        cv.imshow('output',output)
        cv.imshow('input',img)
        k = cv.waitKey(1)

        # key bindings
        if k == 27:         # esc to exit
            break
        elif k == ord('0'): # BG drawing
            print(" mark background regions with left mouse button \n")
            value = DRAW_BG
        elif k == ord('1'): # FG drawing
            print(" mark foreground regions with left mouse button \n")
            value = DRAW_FG
        elif k == ord('2'): # PR_BG drawing
            value = DRAW_PR_BG
        elif k == ord('3'): # PR_FG drawing
            value = DRAW_PR_FG
        elif k == ord('s'): # save image
            bar = np.zeros((img.shape[0],5,3),np.uint8)
            res = np.hstack((img2,bar,img,bar,output))
            cv.imwrite('grabcut_output.png',res)
            print(" Result saved as image \n")
        elif k == ord('r'): # reset everything
            print("resetting \n")
            rect = (0,0,1,1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            value = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
            output = np.zeros(img.shape,np.uint8)           # output image to be shown
        elif k == ord('n'): # segment the image
            print(""" For finer touchups, mark foreground and background after pressing keys 0-3
            and again press 'n' \n""")
            if (rect_or_mask == 0):         # grabcut with rect
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv.GC_INIT_WITH_RECT)
                rect_or_mask = 1
            elif rect_or_mask == 1:         # grabcut with mask
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv.GC_INIT_WITH_MASK)

        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv.bitwise_and(img2,img2,mask=mask2)

    cv.destroyAllWindows()

