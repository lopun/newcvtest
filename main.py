import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
from pytesseract import image_to_string

# Path of working folder on Disk
src_path = "img/"

def get_string(img_path):
    # Read image with opencv
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    img = cv2.resize(img, (0,0), fx=2300/width, fy=2300/width)
    height, width, _ = img.shape
    img = img[0:height, (width // 5):width]

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite(src_path + "removed_noise.png", img)

    threshold_img = img.copy()

    _, threshold_img = cv2.threshold(threshold_img, 80, 255, cv2.THRESH_BINARY)

    #  Apply threshold to get image with only black and white
    threshold_img = cv2.adaptiveThreshold(threshold_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Write the image after apply opencv to do some ...
    cv2.imwrite(src_path + "thres.png", threshold_img)

    circle_img = threshold_img.copy()
    blurred_circle_img = cv2.medianBlur(circle_img, 5)

    circles = cv2.HoughCircles(blurred_circle_img, cv2.HOUGH_GRADIENT, 2, 150, np.array([]), param1=50, param2=60,minRadius=20,maxRadius=80)

    circle_mask = np.full(circle_img.shape, 255.)

    if circles.size > 0:
        a, b, c = circles.shape
        for i in range(b):
            cv2.circle(circle_img, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (255,255,255), 20, cv2.LINE_AA)
            # cv2.circle(circle_img, (circles[0][i][0], circles[0][i][1]), 2, (0,0,0), 1, cv2.LINE_AA)
            cv2.circle(circle_mask, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0,0,0),  -1, 0)

        bytemask = np.asarray(circle_mask, dtype=np.uint8)
        circle_img = cv2.inpaint(circle_img, bytemask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(src_path + "circle.png", circle_img)
        # cv2.imshow("Detected circles", circle_img)
        # cv2.waitKey(0)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(Image.open(src_path + "circle.png"), lang='kor+eng', config='--psm 4 --oem 1').split('\n')

    new_arr = []
    for item in result:
        new_arr += [x for x in item.split(' ')]
    print(new_arr)
    new_arr = [re.sub(r"[^0-9]+","", x) for x in new_arr if re.sub(r"[^0-9]+","", x) != '']
    print(new_arr)

    final_arr = []
    if len(new_arr) == 2:
        calculated = [x for x in new_arr[0].split(' ') if x != '']
        final_arr.append(calculated[0])
        final_arr.append(calculated[1])
        final_arr.append(new_arr[1])
    elif len(new_arr) == 3:
        final_arr = new_arr


    # Remove template file
    #os.remove(temp)

    return result


print('--- Start recognize text from image ---')
get_string(src_path + "test2.png")

print("------ Done -------")
