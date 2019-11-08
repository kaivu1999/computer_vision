import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from random import randrange
import sys
import os

# img_1 = cv2.imread("right.jpg")
# img1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
# img_2 = cv2.imread("left.jpg")
# img2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)

def homo(img1,img2): # left is fixed
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()
    kp1 = sift.detect(img2,None)
    kp2 = sift.detect(img1,None)
    kp1, des1 = sift.compute(img2,kp1)
    kp2, des2 = sift.compute(img1,kp2)
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.75*m[1].distance:
            good.append(m)
    print(len(good))
    # matches = bf.knnMatch(des2,des1,k=2)
    # good1 = []
    # for m in matches:
    #     if m[0].distance < 0.75*m[1].distance:
    #         good1.append(m)
    # print(len(good1))
    # if(len(good) >= len(good1)):
    #     matches = np.asarray(good)
    # else:
    #     matches = np.asarray(good1)
    matches = np.asarray(good)
    if len(matches) >= 4:
        src = np.float32([ kp1[m[0].queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst = np.float32([ kp2[m[0].trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        raise AssertionError('Cant find enough keypoints.')
    return H

#transformed_points = cv2.warpPerspective(p_array, matrix, table_image_size, cv2.WARP_INVERSE_MAP)

def stitch(img2,img1,H): # IMG1 - IMG2
    xlt =   H[0][2]/ H[2][2]
    ylt = H[1][2]/H[2][2]
    xlb = (img1.shape[0]*H[0][1] + H[0][2])/(img1.shape[0]*H[2][1] + H[2][2])
    ylb = (img1.shape[0]*H[1][1] + H[1][2])/(img1.shape[0]*H[2][1] + H[2][2])
    xrt = (img1.shape[1]*H[0][0] + H[0][2])/(img1.shape[1]*H[2][0] + H[2][2])
    yrt = (img1.shape[1]*H[1][0] + H[1][2])/(img1.shape[1]*H[2][0] + H[2][2])
    xrb = (img1.shape[1]*H[0][0] + img1.shape[0]*H[0][1] + H[0][2])/(img1.shape[1]*H[2][0] + img1.shape[0]*H[2][1] + H[2][2])
    yrb = (img1.shape[1]*H[1][0] + img1.shape[0]*H[1][1] + H[1][2])/(img1.shape[1]*H[2][0] + img1.shape[0]*H[2][1] + H[2][2])

    #print(ylt,ylb)
    if(xlt > xrt):
        if(xrt > 0):
            xlt = -(1.5)*img1.shape[1]
            if(xlt > xrt):
                xrt = (1.5)*img1.shape[1] + img2.shape[1]
        else:
            xrt = (1.5)*img1.shape[1] + img2.shape[1]
            if(xlt > xrt):
                xlt = -(1.5)*img1.shape[1]
    if(xlb > xrb):
        if(xrb > 0):
            xlb = -(1.5)*img1.shape[1]
            if(xlb > xrb):
                xrb = (1.5)*img1.shape[1] + img2.shape[1]
        else:
            xrb = (1.5)*img1.shape[1] + img2.shape[1]
            if(xlb > xrb):
                xlb = -(1.5)*img1.shape[1]
    if(ylt > ylb):
        if(ylb<0):
            ylb = (1.5)*img1.shape[0] + img2.shape[0]
            if(ylt > ylb):
                ylt = -(1.5)*img1.shape[0]
        else:
            ylt = -(1.5)*img1.shape[0]
            if(ylt > ylb):
                ylb = (1.5)*img1.shape[0] + img2.shape[0]
    if(yrt > yrb):
        if(yrb<0):
            yrb = (1.5)*img1.shape[0] + img2.shape[0]
            if(yrt > yrb):
                yrt = -(1.5)*img1.shape[0]
        else:
            yrt = -(1.5)*img1.shape[0]
            if(yrt > yrb):
                yrb = (1.5)*img1.shape[0] + img2.shape[0]

    print(yrb,yrt)
    xoff = 0
    yoff = 0
    if(xlt<0 or xlb<0):
        xoff = max(-xlt,-xlb)
        xoff = min(xoff,(1.5)*img1.shape[1])
    if(ylt<0 or yrt<0):
        yoff = max(-ylt,-yrt)
        yoff = min(yoff,(1.5)*img1.shape[0])
    l = xoff + img2.shape[1]
    w = yoff + img2.shape[0]
    if(xrt>img2.shape[1] or xrb>img2.shape[1]):
        temp = max(xrt,xrb)
        temp = min(temp,img2.shape[1]+(1.5)*img1.shape[1])
        l = xoff + temp
    if(yrb>img2.shape[0] or ylb>img2.shape[0]):
        temp = max(ylb,yrb)
        temp = min(temp,img2.shape[0]+(1.5)*img1.shape[0])
        w = yoff + temp

    #print(xoff,yoff)
    H[0][0] += xoff*H[2][0]
    H[0][1] += xoff*H[2][1]
    H[0][2] += xoff*H[2][2]
    H[1][0] += yoff*H[2][0]
    H[1][1] += yoff*H[2][1]
    H[1][2] += yoff*H[2][2]
    dst = cv2.warpPerspective(img1,H,((int)(l), (int)(w)))
    for y in range(0,img2.shape[0]):
        for x in range(0,img2.shape[1]):
            if(img2[y][x][0] != 0 or img2[y][x][1] != 0 or img2[y][x][2] != 0):
                dst[y + (int)(yoff)][x + (int)(xoff)] = img2[y][x]
    return dst

def balance(img2,img1): #img2 is fixed
    (b1,g1,r1) = hist(img1)
    (b2,g2,r2) = hist(img2)
    (b,g,r) = (b2/b1,g2/g1,r2/r1)
    for y in range(0,img1.shape[0]):
        for i  in range(0,img1.shape[1]):
            if b*img1[y][i][0] <= 255:
                img1[y][i][0] *= b
            else:
                img1[y][i][0] = 255
            if g*img1[y][i][1] <= 255:
                img1[y][i][1] *= g
            else:
                img1[y][i][1] = 255
            if r*img1[y][i][2] <= 255:
                img1[y][i][2] *= r
            else:
                img1[y][i][2] = 255
    return

def hist(img):
    (b,g,r) = (0,0,0)
    for y in range(0,img.shape[0]):
        for i  in range(0,img.shape[1]):
            b += img[y][i][0]
            g += img[y][i][1]
            r += img[y][i][2]
    n = img.shape[0]*img.shape[1]
    return (b/n,g/n,r/n)

def good(img1,img2): #img1 fixed
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()
    kp1 = sift.detect(img2,None)
    kp2 = sift.detect(img1,None)
    kp1, des1 = sift.compute(img2,kp1)
    kp2, des2 = sift.compute(img1,kp2)
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.75*m[1].distance:
            good.append(m)
    return len(good)

def ins(l,x,k):
    i = 0
    n = len(l)
    for (x1,k1) in l:
        if(k1<=k):
            l.insert(i,(x,k))
            break
        i += 1
    if(i==n):
        l.append((x,k))
    return

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img, (832,468), interpolation = cv2.INTER_AREA)
        if img is not None:
            images.append(img)
    return images    

# img_0 = cv2.imread("./5/0.jpg")
# img_0 = cv2.resize(img_0, (832,468), interpolation = cv2.INTER_AREA)
# img_1 = cv2.imread("./5/1.jpg")
# img_1 = cv2.resize(img_1, (832,468), interpolation = cv2.INTER_AREA)
# img_2 = cv2.imread("./5/2.jpg")
# img_2 = cv2.resize(img_2, (832,468), interpolation = cv2.INTER_AREA)
# img_3 = cv2.imread("./5/3.jpg")
# img_3 = cv2.resize(img_3, (832,468), interpolation = cv2.INTER_AREA)
# img_4 = cv2.imread("./5/4.jpg")
# img_4 = cv2.resize(img_4, (832,468), interpolation = cv2.INTER_AREA)
# img_5 = cv2.imread("./5/5.jpg")
# img_5 = cv2.resize(img_5, (832,468), interpolation = cv2.INTER_AREA)
# img_6 = cv2.imread("./5/6.jpg")
# img_6 = cv2.resize(img_6, (832,468), interpolation = cv2.INTER_AREA)

img_l = load_images_from_folder(sys.argv[1])
l = []
i=0
for img_ in img_l:
    j=0
    temp = []
    s = 0
    for img_t in img_l:
        if(i!=j):
            g = good(img_,img_t)
            ins(temp,img_t,g)
            s += g
        j += 1
    l.append((img_,s,temp))
    i += 1

gp = 0
for (img_,s,imgl) in l:
    if(s>gp):
        gp = s
        order = imgl
        img_main = img_

print("ordered")
temp_img = img_main
for (img_,g) in order:
    cv2.imwrite("test_all.jpg",temp_img)
    balance(img_main,img_)
    img = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    tempimg = cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
    H = homo(tempimg, img)
    temp_img = stitch(temp_img,img_,H)

cv2.imwrite("test_all.jpg",temp_img)
