"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""

"""Computer Vision - Assignment 1 - Video Processing with OpenCV
   Saptarshi Chakrabarti - r0123456
   19-03-2024
"""

""" python CV_showcase.py -i inputfilm.mp4 -o outputfilm.mp4 """

import cv2 as cv
import numpy as np
import sys
import imutils
import argparse

# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv.CAP_PROP_POS_MSEC)) < upper


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    backSub = cv.createBackgroundSubtractorMOG2()

    # HSV space color ranges
    # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])

    # Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])

    # Green color
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])

    # BGR color ranges
    bgr_low_red = (0, 0, 255)
    bgr_high_red = (125, 125, 255)
    bgr_low_blue = (255, 0, 0)
    bgr_high_blue = (255, 125, 125)
    bgr_low_green = (0, 50, 0)
    bgr_high_green = (200, 255, 200)

    # While loop to process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        subtitle = 'The Trip'
        
        if ret:
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

            # 2.1-----Switch the movie beween color and grayscale a few times (±4s)
            # Switch between color and grayscale
            if between(cap, 500, 4000):
                subtitle = 'Switch between colour and grayscale : Colour'
                if between(cap, 1000, 2000) or between(cap, 3000, 4000):
                    subtitle = 'Switch between colour and grayscale : Grayscale'
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
                    
            # 2.2.1-----Gaussian blurring with different kernel sizes
            switchstep = 1000
            # Gaussian filter with kernels between 3 and 9
            if between(cap, 4000, 8000):
                for i in range(4):
                    beginframe = 4000 + (i * switchstep)
                    endframe = 5000 + (i * switchstep)
                    kernel_size = 3 + (i * 2)
                    if between(cap, beginframe, endframe):
                        subtitle = f'Gaussian filter with kernel ({kernel_size}, {kernel_size})'
                        frame = cv.GaussianBlur(frame, (kernel_size, kernel_size), cv.BORDER_DEFAULT)

            
            # 2.2.2-----Bilateral filter with different parameters
            # Bilateral filter with parameters
            if between(cap, 8000, 12000):
                for i in range(4):
                    beginframe = 8000 + (i * switchstep)
                    endframe = 9000 + (i * switchstep)
                    d = 15 + (i * 5)
                    sigmaColor = 50 + (i * 50)
                    sigmaSpace = 50 + (i * 50)
                    if between(cap, beginframe, endframe):
                        subtitle = f'Bilateral filter with parameters ({d}, {sigmaColor}, {sigmaSpace})'
                        frame = cv.bilateralFilter(frame, d, sigmaColor, sigmaSpace, cv.BORDER_DEFAULT)

            # 2.3.1-----Thresholding of blue-gray colour in BGR Color Space
            if between(cap, 12000, 14000):
                subtitle = 'Thresholding of blue-gray colour in BGR Color Space'
                # frame = cv.inRange(frame, (97,71,68), (159,152,152))
                mask = cv.inRange(frame, (97,71,68), (159,152,152))
                frame = cv.bitwise_and(frame, frame, mask=mask)
                # convert frame to RGB from BGR
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # 2.3.2-----Thresholding of blue colour in HSV Color Space
            if between(cap, 14000, 17000):
                subtitle = 'Thresholding of blue colour in HSV Space'
                frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                # frame = cv.inRange(frame_hsv, low_blue, high_blue)
                mask = cv.inRange(frame_hsv, low_blue, high_blue)
                frame = cv.bitwise_and(frame, frame, mask=mask)
                # convert frame to RGB from BGR
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # 2.3.3----binary morphological operations
            # Thresholding of blue color with dilate + erode
            if between(cap, 17000, 20000):
                subtitle = 'Thresholding of blue colour with dilate + erode in HSV Space'

                # Define the kernel
                kernel = np.ones((9, 9), 'uint8')
                
                # Convert frame to HSV
                frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                
                # Create mask for blue color
                mask_blue = cv.inRange(frame_hsv, low_blue, high_blue)
                
                # Perform dilation and erosion
                mask_dilated = cv.dilate(mask_blue, kernel, iterations=1)
                mask_eroded = cv.erode(mask_dilated, kernel, iterations=1)
                
                # Convert the masks to BGR for visualization
                mask_blue_bgr = cv.cvtColor(mask_blue, cv.COLOR_GRAY2BGR)
                mask_dilated_bgr = cv.cvtColor(mask_dilated, cv.COLOR_GRAY2BGR)
                mask_eroded_bgr = cv.cvtColor(mask_eroded, cv.COLOR_GRAY2BGR)
                
                # Stack the masks horizontally for visualization
                mask_combined = np.hstack((mask_blue_bgr, mask_dilated_bgr, mask_eroded_bgr))
                
                # Display the combined masks
                # cv2.imshow('Combined Masks', mask_combined)
                
                # Superimpose the new mask over the original frame
                frame = cv.addWeighted(frame, 0.5, mask_eroded_bgr, 0.5, 0)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # 3.1
            # Sobel - edge detector -> Vertical lines
            if between(cap, 20000, 22500):
                if between(cap, 20000, 21000):
                    subtitle = 'Sobel - Edge detector - Vertical lines - Kernel size : 3'
                    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    sobel_frame = cv.Sobel(gray_frame, cv.CV_8U, dx=1, dy=0, ksize=3)
                    

                if between(cap, 21000, 22500):
                    subtitle = 'Sobel - Edge detector - Vertical lines - Kernel size : 5'
                    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    sobel_frame = cv.Sobel(gray_frame, cv.CV_8U, dx=1, dy=0, ksize=5)
                
                # Create a canvas with the same size as the frame
                canvas = np.zeros_like(frame)
                canvas[:] = (0, 0, 0)  # Set canvas to black

                # Set the color of the lines to green 
                color = (0, 255, 0)  # Green color

                # Find the edges using Sobel and draw them on the canvas
                edges = cv.threshold(sobel_frame, 50, 255, cv.THRESH_BINARY)[1]
                contour, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(canvas, contour, -1, color, 2)
                frame = cv.addWeighted(canvas, 1, canvas, 1, 0)

                # Overlay the lines on the original frame
                frame = cv.addWeighted(frame, 1, canvas, 1, 0)

            # Sobel - edge detector -> Horizontal lines   
            if between(cap, 22500, 25000):
                if between(cap, 22500, 23500):
                    subtitle = 'Sobel - Edge detector - Horizontal lines - Kernel size : 3'
                    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    sobel_frame = cv.Sobel(gray_frame, cv.CV_8U, dx=0, dy=1, ksize=3)

                if between(cap, 23500, 25000):
                    subtitle = 'Sobel - Edge detector - Horizontal lines - Kernel size : 5'
                    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    sobel_frame = cv.Sobel(gray_frame, cv.CV_8U, dx=0, dy=1, ksize=5)
                
                # Create a canvas with the same size as the frame
                canvas = np.zeros_like(frame)
                canvas[:] = (0, 0, 0)  # Set canvas to black

                # Set the color of the lines to red 
                color = (0, 0, 255)  # Red color

                # Find the edges using Sobel and draw them on the canvas
                edges = cv.threshold(sobel_frame, 50, 255, cv.THRESH_BINARY)[1]
                contour, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(canvas, contour, -1, color, 2)
                frame = cv.addWeighted(canvas, 1, canvas, 1, 0)

                # Overlay the lines on the original frame
                frame = cv.addWeighted(frame, 1, canvas, 1, 0)

            if between(cap, 25000, 26000):
                subtitle = 'Waiting for arrival...'
                
            # Hough transform -> Detect circles with tweaked parameters
            if between(cap, 26000, 30000):
                subtitle = 'Hough circles with appropiate parameters'
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # Apply Gaussian blur to reduce noise
                blurred = cv.GaussianBlur(gray, (9, 9), 0)

                # Compute the gradient using Sobel
                grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
                grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)

                # Compute the gradient magnitude
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

                # Threshold the gradient image
                _, gradient_thresholded = cv.threshold(gradient_magnitude, 50, 255, cv.THRESH_BINARY)

                # Convert to CV_8UC1 if needed
                gradient_thresholded = cv.convertScaleAbs(gradient_thresholded)

                # Apply Hough Circle Transform
                circles = cv.HoughCircles(gradient_thresholded, 
                            cv.HOUGH_GRADIENT,
                            dp=1,          # Inverse ratio of accumulator resolution to image resolution
                            minDist=100,   # Minimum distance between detected centers
                            param1=550,    # Upper threshold for Canny edge detection
                            param2=120,    # Threshold for center detection
                            minRadius=2,   # Minimum radius
                            maxRadius=0    # Maximum radius (0: no limit)
                            )

                # Draw detected circles on the original image
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (x, y, r) in circles:
                        cv.circle(frame, (x, y), r, (0, 255, 0), 2)
                        cv.circle(frame, (x, y), 2, (0, 255, 255), 3)
                
        
            # Hough transform -> Detect circles with inappropiate parameters
            if between(cap, 30000, 35000):
                subtitle = 'Hough circles with inappropiate parameters'
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # Apply Gaussian blur to reduce noise
                blurred = cv.GaussianBlur(gray, (9, 9), 0)

                # Compute the gradient using Sobel
                grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
                grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)

                # Compute the gradient magnitude
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

                # Threshold the gradient image
                _, gradient_thresholded = cv.threshold(gradient_magnitude, 50, 255, cv.THRESH_BINARY)

                # Convert to CV_8UC1 if needed
                gradient_thresholded = cv.convertScaleAbs(gradient_thresholded)

                # Apply Hough Circle Transform
                circles = cv.HoughCircles(gradient_thresholded, 
                            cv.HOUGH_GRADIENT,
                            dp=23,          # Inverse ratio of accumulator resolution to image resolution
                            minDist=10,     # Minimum distance between detected centers
                            param1=50,      # Upper threshold for Canny edge detection
                            param2=10,      # Threshold for center detection
                            minRadius=20,   # Minimum radius
                            maxRadius=200   # Maximum radius (0: no limit)
                            )

                # Draw detected circles on the original image
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (x, y, r) in circles:
                        cv.circle(frame, (x, y), r, (0, 255, 0), 2)
                        cv.circle(frame, (x, y), 2, (0, 255, 255), 3)
                
                
            object = "ship.jpg"

            # Rectangle around object of interest
            if between(cap, 35500, 37000):
                
                subtitle = 'Rectangle around object of interest - the boats.'
                
                img_interest = cv.imread(object, 0) # Image of object to be detected
                # img_interest =  cv.cvtColor(img_interest, cv.COLOR_BGR2GRAY) # turn it into grayscale
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
                
                w,h = img_interest.shape[::-1]

                # Match with squares differences
                temp = cv.matchTemplate(frame_gray, img_interest, cv.TM_SQDIFF)
                
                (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(temp)              
                (startX, startY) = minLoc 
                
                endX = startX + img_interest.shape[1]
                endY = startY + img_interest.shape[0]               
                cv.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    
            # Likelihood of the object of interest in a position
            if between(cap, 37000, 40000):
                subtitle = 'Likelihood of the object of interest in a position'
                img_interest = cv.imread(object, 0) # Image of the object of interest in grayscale
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)               
                w,h = img_interest.shape[::-1]
                temp = cv.matchTemplate(frame_gray, img_interest, cv.TM_SQDIFF)
            
                inv_probability = cv.normalize(temp, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
                probability = cv.bitwise_not(inv_probability) 
                probability = cv.resize(probability, (frame.shape[1], frame.shape[0])) 
                frame = cv.cvtColor(probability, cv.COLOR_GRAY2BGR)
                
            
            # Cartoonify the video
            if between(cap, 40000, 50000):
                subtitle = 'Cartoonify the video'
                frame = cv.stylization(frame, sigma_s=60, sigma_r=0.6)
                
            # detect contours
            if between(cap, 50000, 52000):
                subtitle = 'Detect contours in the video'
                # blurred = cv.GaussianBlur(gray, (3, 3), 0)
                fg_mask = backSub.apply(frame)
                # Find contours
                contours, hierarchy = cv.findContours(fg_mask, 
                                                      cv.RETR_EXTERNAL, 
                                                      cv.CHAIN_APPROX_SIMPLE)
                # print(contours)
                frame = cv.drawContours(frame, contours, -1, (0, 255, 0), 2)
                
            # Remove backgrounds of moving objects
            if between(cap, 52000, 55000):
                subtitle = 'Remove backgrounds of moving objects in the video'
                # blurred = cv.GaussianBlur(gray, (3, 3), 0)
                fg_mask = backSub.apply(frame)
                # Find contours
                contours, hierarchy = cv.findContours(fg_mask, 
                                                      cv.RETR_EXTERNAL, 
                                                      cv.CHAIN_APPROX_SIMPLE)
                # print(contours)
                frame = cv.drawContours(frame, contours, -1, (0, 255, 0), 2)
                # apply global threshold to remove shadows
                _, mask_thresh = cv.threshold(fg_mask, 180, 255, cv.THRESH_BINARY)
                # set the kernel
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
                # Apply erosion
                # frame = cv.morphologyEx(mask_thresh, cv.MORPH_OPEN, kernel)
                mask = cv.morphologyEx(mask_thresh, cv.MORPH_OPEN, kernel)
                # Remove the background
                frame = cv.bitwise_and(frame, frame, mask=mask)
                # convert frame to RGB from BGR
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                
            # Detection and tracking of moving objects
            if between(cap, 55000, 60000):
                if between(cap, 55000, 58000):
                    subtitle = 'Detect and track moving objects'
                else:
                    subtitle = 'Special thanks to Jacob and Zoe for starring in the video!'
                
                # blurred = cv.GaussianBlur(gray, (3, 3), 0)
                fg_mask = backSub.apply(frame)
                # Find contours
                contours, hierarchy = cv.findContours(fg_mask, 
                                                      cv.RETR_EXTERNAL, 
                                                      cv.CHAIN_APPROX_SIMPLE)
                # print(contours)
                # frame = cv.drawContours(frame, contours, -1, (0, 255, 0), 2)

                # apply global threshold to remove shadows
                _, mask_thresh = cv.threshold(fg_mask, 180, 255, cv.THRESH_BINARY)
                # set the kernel
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
                # Apply erosion
                mask_eroded = cv.morphologyEx(mask_thresh, cv.MORPH_OPEN, kernel)

                min_contour_area = 500  # Define your minimum area threshold
                large_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]
                frame_out = frame.copy()
                for cnt in large_contours:
                    x, y, w, h = cv.boundingRect(cnt)
                    frame_out = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
                    frame = frame_out

                
            # Subtitles
            # get boundary of this text
            font = cv.FONT_HERSHEY_SIMPLEX
            textsize = cv.getTextSize(subtitle, font, 0.6, 1)[0]

            # get coords based on boundary
            textX = int((frame_width - textsize[0]) / 2)
            # textY = int((frame_height - textsize[1]) / 2)

            # Create a translucent box to place the subtitle
            x, y, w, h = int((frame_width - textsize[0] - 20)/2), (frame_height - 50), (textsize[0] + 20), (textsize[1] * 2)
            sub_box = frame[y:y+h, x:x+w]
            # white_rect = np.zeros(sub_box.shape, dtype=np.uint8) * 255
            black_rect = np.zeros(sub_box.shape, dtype=np.uint8) * 255
            res = cv.addWeighted(sub_box, 0.5, black_rect, 0.5, 1.0)
            frame[y:y+h, x:x+w] = res
            # Output subtitles -centred in the box
            cv.putText(frame, subtitle, (textX, frame_height - 30), font, 0.6, (255, 255, 255), 1)
                
            # info-text
            id = "r0123456 // Saptarshi Chakrabarti"
            id_textsize = cv.getTextSize(id, font, 0.4, 1)[0]

            # Positioning 20 pixels below the top and 20 pixels to the right
            id_textX = 25
            id_textY = 23 + id_textsize[1]  # Adjusted to leave a small gap

            # Create a translucent box to place the text
            id_x, id_y, id_w, id_h = 20, 20, (id_textsize[0] + 16), (id_textsize[1] * 2)
            id_box = frame[id_y:id_y+id_h, id_x:id_x+id_w]
            id_black_rect = np.zeros(id_box.shape, dtype=np.uint8) * 255
            id_res = cv.addWeighted(id_box, 0.5, id_black_rect, 0.5, 1.0)
            frame[id_y:id_y+id_h, id_x:id_x+id_w] = id_res

            # ID - centred in the box
            cv.putText(frame, id, (id_textX, id_textY), font, 0.4, (255, 255, 255), 1)

            # write frame that you processed to output
            out.write(frame)
            
            # (optional) display the resulting frame
            cv.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)

# End of file