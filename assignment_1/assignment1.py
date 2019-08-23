import argparse
import cv2
import numpy as np
 


def check_end(x,y,frame):
  ans =  False
  w,h = frame.shape
  count = 0
  for i in range(20):
    for j in range(100):
      if x-10+i < w and y-50+j < h:
        if(frame[x-10+i,y-50+j] > 100):
          count+=1
  if count <= 1:
    ans = True
  return ans
  
def main(args):
  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  cap = cv2.VideoCapture(args.path)
  # Define the codec and create VideoWriter object
  # fourcc = cv2.VideoWriter_fourcc(*'XVID')
  # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  # fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
  # out = cv2.VideoWriter('output.mp4', 0x00000021, 15.0, (1280,360))
  out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  fgbg = cv2.createBackgroundSubtractorMOG2()

  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")
  
  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
      fgmask = fgbg.apply(frame) 
      fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
      
      # opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
      
      edges = cv2.Canny(fgmask,100,200)

      minLineLength = 100
      maxLineGap = 20

      n_white_pix = np.sum(fgmask == 255)
      n_white_pix/= 200
      # print('Number of white pixels:', n_white_pix)

      lines = cv2.HoughLines(edges,1,np.pi/180,60)
      if lines is not None:
          # if len(lines) >= 2:
          #     x1 = 0 
          #     y1 = 0 
          #     x2 = 0 
          #     y2 = 0 
          #     x_1 = 0 
          #     y_1 = 0 
          #     x_2 = 0 
          #     y_2 = 0
          #     for rho, theta in lines[0]:
          #       a = np.cos(theta)
          #       b = np.sin(theta)
          #       x0 = a*rho
          #       y0 = b*rho
          #       x1 = int(x0 + 100*(-b))
          #       y1 = int(y0 + 100*(a))
          #       x2 = int(x0 - 200*(-b))
          #       y2 = int(y0 - 200*(a))
          #     for rho, theta in lines[1]:
          #       a = np.cos(theta)
          #       b = np.sin(theta)
          #       x0 = a*rho
          #       y0 = b*rho
          #       x_1 = int(x0 + 100*(-b))
          #       y_1 = int(y0 + 100*(a))
          #       x_2 = int(x0 - 200*(-b))
          #       y_2 = int(y0 - 200*(a))

          #     x1 = int ((x1 + x_1) / 2)
          #     y1 = int ((y1 + y_1) / 2)
          #     y2 = int ((y2 + y_2) / 2)
          #     x2 = int ((x2 + x_2) / 2)
          #     if theta > 1.5:
          #       n_white_pix*=2
          #       x_temp = int(x1 - n_white_pix*(-b))
          #       y_temp = int(y2 - n_white_pix*(a))
          #       x2 = x_temp
          #       y2 = y_temp
          #     else:
          #       x_temp = int(x1 + n_white_pix*(-b))
          #       y_temp = int(y1 + n_white_pix*(a))
          #       x1 = x_temp
          #       y1 = y_temp
          #     cv2.circle(frame,(x1, y1), 20, (0,255,0), -1)
          #     cv2.circle(frame,(x2, y2), 20, (255,0,0), -1)
          #     # cv2.circle(frame,(x0, y0), 20, (255,255,0), -1)
          #     cv2.circle(frame,(0, 0), 20, (255,255,0), -1)
          #     cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),3)
          #     x1 = 0
          #     y1 = 0
          #     x2 = 0
          #     y2 = 0
          #     x_1 = 0
          #     y_1 = 0
          #     x_2 = 0
          #     y_2 = 0
          #     for rho, theta in lines[0]:
          #         a = np.cos(theta)
          #         b = np.sin(theta)
          #         x0 = a*rho
          #         y0 = b*rho
          #         x1 = int(x0 + n_white_pix*(-b))
          #         y1 = int(y0 + n_white_pix*(a))
          #         x2 = int(x0 - n_white_pix*(-b))
          #         y2 = int(y0 - n_white_pix*(a))
          #         # cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),1)
          #     for rho, theta in lines[1]:
          #         a = np.cos(theta)
          #         b = np.sin(theta)
          #         x0 = a*rho
          #         y0 = b*rho
          #         x_1 = int(x0 + 1000*(-b))
          #         y_1 = int(y0 + 1000*(a))
          #         x_2 = int(x0 - 1000*(-b))
          #         y_2 = int(y0 - 1000*(a))
          #         # cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),1)
          #     cv2.line(frame,(int((x1+x_1)/2),int((y1+y_1)/2)),(int((x2+x_2)/2),int((y2+y_2)/2)),(0,0,255),3)
        
          for rho, theta in lines[0]:
              a = np.cos(theta)
              b = np.sin(theta)
              x0 = a*rho
              y0 = b*rho
              x1 = int(x0 + 100*(-b))
              y1 = int(y0 + 100*(a))
              x2 = int(x0 - 200*(-b))
              y2 = int(y0 - 200*(a))

              if theta > 1.6:
                n_white_pix*=2
                x_temp = int(x2 - n_white_pix*(-b))
                y_temp = int(y2 - n_white_pix*(a))
                x2 = x_temp
                y2 = y_temp
              else:
                x_temp = int(x1 + n_white_pix*(-b))
                y_temp = int(y1 + n_white_pix*(a))
                x1 = x_temp
                y1 = y_temp
              # cv2.circle(frame,(x1, y1), 20, (0,255,0), -1)
              # cv2.circle(frame,(x2, y2), 20, (255,0,0), -1)
              cv2.line(frame,(x1,y1),(x2,y2),(130,30,255),3)
   
      cv2.imshow('Frame',frame)
      out.write(frame)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25 ) & 0xFF == ord('q'):
        break
  
    # Break the loop
    else: 
      break
  
  # When everything done, release the video capture object
  cap.release()
  
  # Closes all the frames
  cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'input the file name of the video')
    parser.add_argument('path', help='Add the path to the video',default="./1.mp4")
    args = parser.parse_args()

    main(args)