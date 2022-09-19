import numpy as np
import cv2 as cv
import math
import time
from matplotlib import pyplot as plt

sift = cv.SIFT_create()
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
fast = cv.FastFeatureDetector_create()
orb = cv.ORB_create()

def img_to_kp(img):
    keypoints = np.argwhere(img > 0.01 * img.max())
    keypoints = [cv.KeyPoint(float(x[1]), float(x[0]), 1) for x in keypoints]
    return keypoints

def canny_kp(img):
    dst = cv.Canny(img,100,200)
    return img_to_kp(dst)

def harris_kp(img):
    dst = cv.cornerHarris(img,2,3,0.04)
    return img_to_kp(dst)

def brief_disparity(kp1,kp2,img1,img2):
    kp1,des1 = brief.compute(img1,kp1)
    kp2,des2 = brief.compute(img2,kp2)

    bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches,key=lambda x:x.distance)

    return calc_disparity(kp1,kp2,matches,img1.shape)

def canny_stereo(img1,img2):
    kp1 = canny_kp(img1)
    kp2 = canny_kp(img2)
    return brief_disparity(kp1,kp2,img1,img2)

def canny_harris_stereo(img1,img2):
    kp1 = canny_kp(img1)
    kp2 = canny_kp(img2)
    kp1+= harris_kp(img1)
    kp2+= harris_kp(img2)
    return brief_disparity(kp1,kp2,img1,img2)

def sift_fast_kp(img):
    kp = sift.detect(img,None)
    kp += fast.detect(img,None)
    return kp

def sift_fast_stereo(img1,img2,n):
    kp1 = sift_fast_kp(img1)
    kp2 = sift_fast_kp(img2)
    if n == 0:
        kp1,des1 = sift.compute(img1,kp1)
        kp2,des2 = sift.compute(img2,kp2)
    else:
        kp1,des1 = brief.compute(img1,kp1)
        kp2,des2 = brief.compute(img2,kp2)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return calc_disparity(kp1,kp2,matches,img1.shape)

def canny_sift_fast_stereo(img1,img2):
    kp1 = canny_kp(img1)
    kp2 = canny_kp(img2)

    kp1 += sift_fast_kp(img1)
    kp2 += sift_fast_kp(img2)

    return brief_disparity(kp1,kp2,img1,img2)

def canny_orb_stereo(img1,img2):
    kp1 = list(orb.detect(img1))
    kp2 = list(orb.detect(img2))
    
    kp1 += canny_kp(img1)
    kp2 += canny_kp(img2)

    kp1,des1 = orb.compute(img1,kp1)
    kp2,des2 = orb.compute(img2,kp2)

    bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches,key=lambda x:x.distance)

    return calc_disparity(kp1,kp2,matches,img1.shape)  

def opencv_stereo(img1,img2):
    stereo = cv.StereoBM_create(numDisparities=16,blockSize=15)
    oc_disp = stereo.compute(img1,img2)
    oc_disp = cv.convertScaleAbs(oc_disp)
    return oc_disp

def time_stereo(stereo_f,args):
    start = time.time()
    img = stereo_f(*args)
    end = time.time()
    return img, end-start

def calc_MSE(img1, img2):
        squared_diff = (img1 - img2) ** 2
        summed = np.sum(squared_diff)
        num_pix = img1.shape[0] * img1.shape[1]
        err = summed / num_pix
        return err

def calc_disparity(kp1,kp2,matches,img_shape):
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches] 
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

    disp = np.zeros(img_shape, dtype=np.uint8)

    i = 0
    for k in list_kp1:
        if disp[int(k[1]),int(k[0])] == 0:
            dx2 = (list_kp2[i][1]-k[1])**2         
            dy2 = (list_kp2[i][0]-k[0])**2         
            distance = math.sqrt(dx2 + dy2)
            disp[int(k[1]),int(k[0])] = distance * 8; 
        i += 1
    return disp

def myimshow(img,label,r,c,i):
    plt.subplot(r,c,i)
    plt.axis('off')
    plt.title(label)
    plt.imshow(img,cmap='gray',vmin=0,vmax=255)

def test_disparity(folder_name):
    path1,path2,path_gt = im2_str(folder_name)
    img1 = cv.imread(path1,cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(path2,cv.IMREAD_GRAYSCALE)
    img_gt = cv.imread(path_gt,cv.IMREAD_GRAYSCALE)

    oc_disp,oc_t = time_stereo(opencv_stereo,(img1,img2))
    # disparity calculations of my methods
    canny_disp,canny_t = time_stereo(canny_stereo, (img1,img2))
    ch_disp,ch_t = time_stereo(canny_harris_stereo,(img1,img2))
    sf_disp,sf_t = time_stereo(sift_fast_stereo,(img1,img2,0))
    sfb_disp,sfb_t = time_stereo(sift_fast_stereo,(img1,img2,1))
    csf_disp,csf_t = time_stereo(canny_sift_fast_stereo,(img1,img2))
    co_disp,co_t = time_stereo(canny_orb_stereo,(img1,img2))

    # Error calculations
    print(f"\n\nMean Squared Errors and Execution Times for {folder_name}")
    print("====================")
    print(f"OpenCV: {calc_MSE(oc_disp,img_gt):.6f}, {oc_t:.5f}s")
    print(f"Canny: {calc_MSE(canny_disp,img_gt):.6f}, {canny_t:.5f}s")
    print(f"Canny+Harris: {calc_MSE(ch_disp,img_gt):.6f}, {ch_t:.5f}s")
    print(f"Sift+Fast: {calc_MSE(sf_disp,img_gt):.6f}, {sf_t:.5f}s")
    print(f"Sift+Fast w/BRIEF: {calc_MSE(sfb_disp,img_gt):.6f}, {sfb_t:.5f}s")
    print(f"Canny+Sift+Fast: {calc_MSE(csf_disp,img_gt):.6f}, {csf_t:.5f}s")
    print(f"Canny+ORB: {calc_MSE(co_disp,img_gt):.6f}, {co_t:.5f}s")

    row = 2
    col = 4
    myimshow(img_gt,"ground truth",row,col,1)
    myimshow(oc_disp,"opencv stereo",row,col,2)
    myimshow(canny_disp,"canny",row,col,3)
    myimshow(ch_disp,"canny+harris",row,col,4)
    myimshow(sf_disp,"sift+fast",row,col,5)
    myimshow(sfb_disp,"sift+fast w/brief",row,col,6)
    myimshow(csf_disp,"canny+sift+fast",row,col,7)
    myimshow(co_disp,"canny+ORB",row,col,8)

    plt.show()

def im2_str(folder):
    return f"img/{folder}/im2.ppm",f"img/{folder}/im6.ppm",f"img/{folder}/disp2.pgm"

test_disparity("barn1")
test_disparity("barn2")
test_disparity("bull")
test_disparity("poster")
test_disparity("sawtooth")
test_disparity("venus")
     
