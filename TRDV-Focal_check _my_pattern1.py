from imutils import perspective
from imutils import rotate
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import pandas as pd
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from os import listdir
from sklearn.metrics import auc
from sklearn.linear_model import LinearRegression


root = Tk()
root.title('TRIMBLE_FOCAL CHECK')
# imag_path1 = 'C:/Users/Calibration/Desktop/OPTIC/Optics_softwares/GUI_IMG/TreadView-Header.jpg'
# imag_path2 = 'C:/Users/Calibration/Desktop/OPTIC/Optics_softwares/GUI_IMG/TreadView-Outputs-1.jpg'
imag_path1 = 'D:/Trimble/TRDV_focal_Check/Pycharm/GUI_IMG/TreadView-Header.jpg'
imag_path2 = 'D:/Trimble/TRDV_focal_Check/Pycharm/GUI_IMG/TreadView-Outputs-1.jpg'

image1 = Image.open(imag_path1)
image2 = Image.open(imag_path2)
photo = ImageTk.PhotoImage(image1)
photo2 = ImageTk.PhotoImage(image2)
label = Label(image=photo)
label.image = photo  # keep a reference!
label.pack()

def focal_check_end(thresh_factor=0.1, HowManyContour=3, dilalation_iter=2, erod_iter=1, chunk_num=18):
    img_path = filedialog.askopenfilename(title='Choose the image')
    img = cv2.imread(img_path)
    height,width,channel = img.shape
    img1 = img.copy()


    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img[:,:width//100]=0
    gray_img[:, 99*width // 100:] = 0

    # plt.imshow(gray_img)
    # plt.show()


    blur = cv2.GaussianBlur(src=gray_img, ksize=(5, 5), sigmaX=0)


        # opened = cv2.morphologyEx(blur, cv2.MORPH_OPEN, np.ones((15, 15)))
        # plt.imshow(opened)
        # plt.show()
    _,dst = cv2.threshold(src=blur, thresh=(img.max()) * thresh_factor, maxval=255, type=cv2.THRESH_BINARY)

    edge = cv2.Canny(image=dst, threshold1=0, threshold2=130)
    dialated = cv2.dilate(src=edge, kernel=np.ones((1,5), np.uint8), iterations=dilalation_iter)
    # plt.imshow(dialated)
    # plt.show()
    dialated = cv2.dilate(src=dialated, kernel=(10, 10),anchor=(0,0) ,iterations=dilalation_iter)
    # plt.imshow(dialated)
    # plt.show()
    dialated = cv2.dilate(src=dialated, kernel=(10, 10), anchor=(-1, -1), iterations=dilalation_iter)
    # plt.imshow(dialated)
    # plt.show()

    # eroded = cv2.erode(src=dialated, kernel=(3, 3), iterations=erod_iter)
    # plt.figure(figsize=(12, 12))
    # plt.subplot(321)
    # plt.title('gray')
    # plt.imshow(gray_img)
    # plt.subplot(322)
    # plt.title('blur')
    # plt.imshow(blur)
    # plt.subplot(323)
    # plt.title('thresh')
    # plt.imshow(dst)
    # plt.subplot(324)
    # plt.title('edge')
    # plt.imshow(edge)
    # plt.subplot(325)
    # plt.title('dialated')
    # plt.imshow(dialated)
    # plt.subplot(326)
    # # plt.title('eroded')
    # # plt.imshow(eroded)
    # plt.tight_layout()
    # plt.show()
    contours, hierarchy = cv2.findContours(image=dialated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    area = [cv2.contourArea(i) for i in contours]
    area.sort(reverse=True)
    area = area[:HowManyContour]
    corners = []
    mask = []

    for i, j in enumerate(contours):
        if cv2.contourArea(j) in area:
            rectangle = cv2.minAreaRect(j)
            points = cv2.boxPoints(rectangle)
            points = perspective.order_points(points)
            black_background = np.zeros(gray_img.shape)
            cv2.drawContours(black_background, contours, i, 255, -1)

            corners.append(points)
                # plt.figure(figsize=(12,12))
                # plt.imshow(black_background)
                # plt.show()
                # # name = img_path+'\\'+kk
                # name = name.replace('.tif',"")
            mask.append(black_background)


        else:
             continue
    for corner in corners:
        x1, y1, x2, y2, x3, y3, x4, y4 = corner.ravel()
        cv2.line(img=img1, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
        cv2.line(img=img1, pt1=(x2, y2), pt2=(x3, y3), color=(255, 255, 0), thickness=2)
        cv2.line(img=img1, pt1=(x3, y3), pt2=(x4, y4), color=(0, 255, 255), thickness=2)
        cv2.line(img=img1, pt1=(x4, y4), pt2=(x1, y1), color=(255, 255, 255), thickness=2)
    # plt.imshow(img1)
    # plt.show()
    mask = [cv2.dilate(i,(3,3),anchor=(0,0),iterations=5) for i in mask]
    mask = [cv2.dilate(i, (5, 5), anchor=(-1, -1), iterations=5) for i in mask]
    # for i, j in enumerate(mask):
    #     plt.imsave(
    #         'D:/Trimble/TRDV_focal_Check/Pycharm/result/' + str(
    #             i) + 'mask.png', j)

    image_seg=[cv2.bitwise_and(np.array(i,dtype=np.float32),np.array(gray_img,dtype=np.float32)) for i in mask]
    # for i , j in enumerate(image_seg):
    #     plt.imsave('D:/Trimble/TRDV_focal_Check/Pycharm/result/'+str(i)+'90.png',j)
    overlap=[]
    for i ,j in enumerate(image_seg):
        overlap.append(cv2.addWeighted(j,1,mask[i],0.5,0,dtype = cv2.CV_32F))
    #for i , j in enumerate(overlap):
        #plt.imsave('D:/Trimble/TRDV_focal_Check/Pycharm/result/' + str(i)+'.png',j)

        # plt.imshow(i)
        # plt.show()


    roi = []
    for i, corner in enumerate(corners):
        img1 = image_seg[i].copy()
        x1, y1, x2, y2, x3, y3, x4, y4 = corner.ravel()
        cv2.line(img=img1, pt1=(x1, y1), pt2=(x2, y2), color=255, thickness=1)
        cv2.line(img=img1, pt1=(x2, y2), pt2=(x3, y3), color=255, thickness=1)
        cv2.line(img=img1, pt1=(x3, y3), pt2=(x4, y4), color=255, thickness=1)
        cv2.line(img=img1, pt1=(x4, y4), pt2=(x1, y1), color=255, thickness=1)

        roi_x_min = np.int_(min(x1, x2, x3, x4))
        roi_y_min = np.int_(min(y1, y2, y3, y4))
        roi_x_max = np.int_(max(x1, x2, x3, x4))
        roi_y_max = np.int_(max(y1, y2, y3, y4))
        roi.append(image_seg[i][roi_y_min:roi_y_max, roi_x_min:roi_x_max])


    focal=[]

    for roi1 in roi:
        focal_roi=[]
        for i in range(chunk_num):
            if i + 1 < chunk_num:

                chunk = np.int_(np.linspace(0, roi1.shape[1], chunk_num))
                roi_chunk = roi1[:, chunk[i]:chunk[i + 1]]
                # plt.imshow(roi_chunk)
                # plt.show()
                roi_height,roi_width =  roi_chunk.shape
                roi_chunk[:,:roi_width//100]=0
                roi_chunk[:, 99*roi_width // 100:] = 0
                roi_blured = cv2.GaussianBlur(src=roi_chunk, ksize=(3, 3), sigmaX=0)
                plt.title('roi_chunk')
                # plt.imshow(roi_chunk)
                # plt.show()

                _, roi_dst = cv2.threshold(src=roi_blured, thresh=10, maxval=255, type=cv2.THRESH_BINARY)
                plt.title('roi_thresh')
                # plt.imshow(roi_dst)
                # plt.show()
                roi_dst = roi_dst.astype(np.uint8)

                roi_edge = cv2.Canny(image=roi_dst, threshold1=0, threshold2=130)
                plt.title('roi_edge')
                # plt.imshow(roi_edge)
                # plt.show()
                roi_dialated = cv2.dilate(src=roi_edge, kernel=np.ones((1, 3), np.uint8), iterations=dilalation_iter)
                roi_contours, roi_hierarchy = cv2.findContours(image=roi_dialated, mode=cv2.RETR_EXTERNAL,
                                                               method=cv2.CHAIN_APPROX_SIMPLE)

                roi_area = [cv2.contourArea(i) for i in roi_contours]
                roi_area.sort(reverse=True)
                roi_area = roi_area[:1]
                roi_corners = []


                for i, j in enumerate(roi_contours):
                    if cv2.contourArea(j) in roi_area:
                        rectangle = cv2.minAreaRect(j)
                        roi_points = cv2.boxPoints(rectangle)
                        roi_points = perspective.order_points(roi_points)
                        roi_black_background = np.zeros(roi_chunk.shape)
                        # cv2.drawContours(roi_black_background, contours, i, 255, -1)

                        roi_corners.append(roi_points)
                        plt.figure(figsize=(10,10))

                        # # name = img_path+'\\'+kk
                        # name = name.replace('.tif',"")



                    else:
                        continue
                for corner in roi_corners:
                    x1, y1, x2, y2, x3, y3, x4, y4 = corner.ravel()
                    # cv2.line(img=img1, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
                    cv2.line(img=roi_black_background, pt1=(x2, y2), pt2=(x3, y3), color=(255, 255, 0), thickness=2)
                    thickness = dist.euclidean((x2, y2), (x3, y3))
                    # thickness2 = dist.euclidean((x4, y4), (x1, y1))
                    # cv2.line(img=img1, pt1=(x3, y3), pt2=(x4, y4), color=(0, 255, 255), thickness=2)
                    # cv2.line(img=img1, pt1=(x4, y4), pt2=(x1, y1), color=(255, 255, 255), thickness=2)
                    # plt.imshow(roi_black_background)
                    # plt.show()




                focus = thickness
                focal_roi.append(focus)
        focal.append(focal_roi)
    df = pd.DataFrame(np.array(focal).T, columns=['beam1', 'beam2', 'beam3'])

    df['Chunk_number'] = np.linspace(1, chunk_num-1, chunk_num-1)
    X = np.array(df['Chunk_number']).reshape(-1,1)
    y1 = df['beam1']
    y2 = df['beam2']
    y3 = df['beam3']

    model = LinearRegression()
    model.fit(X,y1)
    # print(f"model coef:{model.coef_} , model intercept:{model.intercept_}")
    model.fit(X, y2)
    # print(f"model coef:{model.coef_} , model intercept:{model.intercept_}")
    model.fit(X, y3)
    # print(f"model coef:{model.coef_} , model intercept:{model.intercept_}")


    # print(focal)
    # df.to_excel('C:/Users/Calibration/Desktop/OPTIC/Optics_softwares/TRDV-FOCAL-CHECK/production_img_result/focal.xlsx')

    for i , j  in enumerate( focal):
        plt.grid()

        plt.subplot(len(focal),1,i+1)
        plt.title('Focal Distribution Of Beam' + str(i + 1))
        plt.plot(j)
    plt.savefig(img_path[:-3] + 'pdf' )


label2 = Label(root, text='Prepared By Optics Department', font=("Times New Roman", 10)).pack(side='bottom',
                                                                                                  fill='both',
                                                                                                  expand='no')
buttom1 = tk.Button(root, text='Choose The image', padx=40, pady=30, command=focal_check_end, font=(14),
                        background='khaki').pack(expand='yes')
root.geometry('1280x800')
root.mainloop()
