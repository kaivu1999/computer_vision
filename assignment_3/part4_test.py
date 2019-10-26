import argparse
import cv2
import numpy as np
import math
import os
from objloader_simple import *
import codecs, json


obj_text = codecs.open("reference/mtx.json", 'r', encoding='utf-8').read()
ob = json.loads(obj_text)
mtx = np.array(ob)

# chess 
square_size = 2.615
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp = objp * square_size
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

square_size = 2.615

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 10  



def main():
    """
    This functions loads the target surface image,
    """
    homography = None

    camera_parameters = mtx # got after doing the caliberation
    # camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # create ORB keypoint detector
    sift = cv2.xfeatures2d.SIFT_create()
    # create BFMatcher object based on hamming distance  
    bf = cv2.BFMatcher()
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    marker1 = cv2.imread(os.path.join(dir_name, 'reference/markers/marker1.jpg'), 0)
    marker1_inverse = cv2.imread(os.path.join(dir_name, 'reference/markers/marker1_inverse.jpg'), 0)
    # Compute marker keypoints and its descriptors
    kp_marker1 = sift.detect(marker1,None)
    kp_marker1, des_marker1 = sift.compute(marker1,kp_marker1)

    kp_marker1_inverse = sift.detect(marker1_inverse,None)
    kp_marker1_inverse, des_marker1_inverse = sift.compute(marker1_inverse,kp_marker1_inverse)

    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)  
    # init video capture

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./reference/videos/test_1.mp4")

    prev5 = np.ones((3,3))
    prev4 = np.ones((3,3))
    prev3 = np.ones((3,3))
    prev2 = np.ones((3,3))
    prev1 = np.ones((3,3))
    homography = np.ones((3,3))

    prev_5 = np.ones((3,3))
    prev_4 = np.ones((3,3))
    prev_3 = np.ones((3,3))
    prev_2 = np.ones((3,3))
    prev_1 = np.ones((3,3))
    homography_2 = np.ones((3,3))

    speed = 10
    Identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
    unit_translation = np.array([[0,0,0],[0,0,1],[0,0,0]])
    prev_trans = np.array([[0,0,0],[0,0,1],[0,0,0]])

    center1 = np.array([0,0])


    n_frame = 0
    inverse = False
    
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while True:

        
        n_frame += 1
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print ("Unable to capture video")
            return 


        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret, corners_c = cv2.findChessboardCorners(gray, (9,6),None)
        # objpoints.append(objp)
        if ret == True:
            h,w = 6,9
            corners_chess = cv2.cornerSubPix(gray,corners_c,(11,11),(-1,-1),criteria)
            # imgpoints.append(corners_chess)
            homography_chess, mask = cv2.findHomography(objp, corners_chess, cv2.RANSAC, 5.0)
            pts_chess = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            pts_chess = pts_chess * square_size
            dst_chess = cv2.perspectiveTransform(pts_chess, homography_chess)
            frame = cv2.polylines(frame, [np.int32(dst_chess)], True, 200, 3, cv2.LINE_AA)  
            # print(homography_chess)
            # Draw and display the corners
            frame = cv2.drawChessboardCorners(frame, (9,6), corners_chess,ret)
            if(point_inside(center1,dst_chess)):
                cv2.waitKey(2000)
                # speed *= -1
                print("Reached destination !!")
                inverse = not inverse
                print("What to do ????")
                prev_trans*=-1
                print("Better I go back ...")
            

        if inverse:
            desMark1 = des_marker1_inverse
            kpMark1 = kp_marker1_inverse
        else:
            desMark1 = des_marker1
            kpMark1 = kp_marker1

        # find and draw the keypoints of the frame
        kp_frame = sift.detect(frame,None)
        kp_frame, des_frame = sift.compute(frame,kp_frame)
        matches1 = bf.knnMatch(desMark1,des_frame, k=2)
        # match frame descriptors with model descriptors
        # sort them in the order of their distance
        # the lower the distance, the better the matc h

        good = []
        for m in matches1:
            if m[0].distance < 0.75*m[1].distance:
                good.append(m)
        matches1 = np.asarray(good)
        # print(len(matches))

        # compute Homography if enough matches are found
        if len(matches1) > MIN_MATCHES:
            # differenciate between source points and destination points
            src_pts = np.float32([kpMark1[m[0].queryIdx].pt for m in matches1]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m[0].trainIdx].pt for m in matches1]).reshape(-1, 1, 2)
            # compute Homography
            
            prev5 = prev4
            prev4 = prev3
            prev3 = prev2
            prev2 = prev1
            prev1 = homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            try:
                avg_homography = ( prev1 + prev2  + prev3 + prev4 + prev5 + homography ) / 6.0 
            except:
                continue
            # avg_homography = homography

            if True:
                # Draw a rectangle that marks the found model in the frame
                h, w = marker1.shape
                
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                center = (pts[0] + pts[1] + pts[2] + pts[3]) / 4
                # project corners into frame
                dst1 = cv2.perspectiveTransform(pts, avg_homography)
                # connect them with lines  
                frame = cv2.polylines(frame, [np.int32(dst1)], True, 255, 3, cv2.LINE_AA)  
            # if a valid homography matrix was found render cube on model plane
            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    # avg_homography = np.matmul(Identity,avg_homography)
                    avg_homography = np.matmul(avg_homography , Identity + prev_trans + unit_translation*speed )
                    prev_trans =  prev_trans + unit_translation*speed
                    dst1 = cv2.perspectiveTransform(pts, avg_homography)
                    center1 = (dst1[0] + dst1[1] + dst1[2] + dst1[3]) / 4 # img coordinates
                    frame = cv2.polylines(frame, [np.int32(dst1)], True, 255, 3, cv2.LINE_AA)
                    # frame = cv2.circle(frame, [np.int32(center)], True, 255, 3, cv2.LINE_AA)
                    projection = projection_matrix(camera_parameters, avg_homography)  
                    # project cube or model
                    frame = render(frame, obj, projection, marker1, False)

                    #frame = render(frame, model, projection)
                except Exception as e: print(e)
            cv2.imshow('window', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break            # draw first 10 matches1.

        else:
            print ("Not enough matches found - %d/%d" % (len(matches1), MIN_MATCHES))
        
    cap.release()
    cv2.destroyAllWindows()
    return 0

def point_inside(pt, square):
    pt1 = square[0]
    pt2 = square[1]
    pt3 = square[2]
    pt4 = square[3]
    tuple3s = [(pt1,pt2,pt3),(pt2,pt3,pt4),(pt3,pt4,pt1),(pt4,pt1,pt2)]
    ans = True
    for tuple3 in tuple3s:
        ans = ans and onCorrectSideOfLine(pt,tuple3)
    return ans

def onCorrectSideOfLine(pt,tuple3): # Line of pt1 and pt2 with pt3 as line decding
    # print(tuple3 , "hi")
    pt1,pt2,pt3 = tuple3
    a = (  ((pt3[0][0] - pt1[0][0]) * (pt3[0][1] - pt2[0][1]))  - ((pt3[0][0] - pt2[0][0]) * (pt3[0][1] - pt1[0][1]))  )
    b = ( ((pt[0][0] - pt1[0][0]) * (pt[0][1] - pt2[0][1])) - ((pt[0][0] - pt2[0][0]) * (pt[0][1] - pt1[0][1])))
    # print(a *  b)
    if a*b > 0:
        return True
    # print(val)
    return False

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')


args = parser.parse_args()

if __name__ == '__main__':
    main()
