import numpy as np
# we import numpy for numerical computation
import cv2
# we import cv2 for open cv functions
import glob
# we import glob for pattern matching
import os
# we import os for operating system related functions

# let's load the image file that found in the directory
for files in glob.iglob('input-test-images/*.jpg'):
     # let's read the image
     originalImage = cv2.imread(files)

     # we convert the image to grayscale
     grayscaleImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

     # let's compute the Scharr gradient magnitude representation of the images in both the x and y direction
     # we used cv2 sobel function with dx and dy parameteres by specifying the gradient direction.
     # the ksize value indicates the use of the schar filter. we assigned the gradient representation values to gradX and gradY variable
     gradientX = cv2.Sobel(grayscaleImage, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
     gradientY = cv2.Sobel(grayscaleImage, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

     # we subtract the y-gradient from the x-gradient to compute the gradient magnitude of the image
     gradientImage = cv2.subtract(gradientX, gradientY)
     # we use converScaleAbs methods of cv2 to convert the gradient image to the absolute representation by
     # taking the absolute value of each pixel and scaling it to the 8-bit range (0-255).
     gradientImage = cv2.convertScaleAbs(gradientImage)

     # we perform blur and threshold operation on the image
     # cv2 blur method applied a blur filter to smooth the image
     blurredImage = cv2.blur(gradientImage, (3, 3))
     # cv2 threshold method converts the blurred image into binary image based on  a specified threshold range
     (_, thresh) = cv2.threshold(blurredImage, 210, 250, cv2.THRESH_BINARY)

     # we construct a closing kernel and apply it to the thresholded image
     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
     closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

     # we perform a series of erosions and dilations operation to enhance and refine the binary image.
     # the cv2 erode function erodes the image by removing boundary pixels
     # the cv2 dilate function dilates the eroded image by adding pixels to the boundaries.
     closed = cv2.erode(closed, None, iterations = 7)
     closed = cv2.dilate(closed, None, iterations = 2)


     # let's find the contours in the thresholded image
     (contours, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

     #  let's sort the contours in descending order based on their areas by using cv 2 contour area method
     c = sorted(contours, key = cv2.contourArea, reverse = True)[0]
     # let's compute the minimum area rectangle that tightly encloses the contour c by using cv 2 minimum Area Rectangle method
     rectangleArea = cv2.minAreaRect(c)
     # we used the cv 2 box points method to obtain the four corner coordinates of the rectangle as floating-point values.
     box = np.int0(cv2.boxPoints(rectangleArea))
     # let's draw the contour of the minimum area rectangle on the original image.

     cv2.drawContours(originalImage, [box], -1, (0, 255, 0), 3)
     # let's display the image finally
     cv2.imshow("Image", originalImage)
     cv2.waitKey(0)

     detectedImageFile = 'output-test-images/' + os.path.basename(files);
     cv2.imwrite(detectedImageFile, originalImage)
     
