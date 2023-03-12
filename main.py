import math

import numpy as np
import cv2
import glob

from EniPy import colors
from EniPy import eniUtils

def getScaledImage(image, targetWidth = 1920):
    scale = targetWidth / image.shape[1]
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    scaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return scaled

def loadImage(filename):
    targetWidth = 1080
    image = cv2.imread(filename)

    scale = targetWidth / image.shape[1]
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    #image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image

def getBlankImage(width, height, color = colors.Black):
    blankImage = np.zeros((height, width, 3), np.uint8)
    blankImage[:] = color
    return blankImage
def create_collages(images):

    collage_size = int(math.ceil(math.sqrt(len(images))))

    collage = None

    if len(images) > 0:
        targetWidth = images[0].shape[1]
        targetHeight = images[0].shape[0]

    for colIndex in range(collage_size):
        row = None
        for rowIndex in range(collage_size):
            i = colIndex * collage_size + rowIndex
            insertFrame = getBlankImage(targetWidth, targetHeight, colors.White)
            if i < len(images):
                insertFrame = images[i]

            if row is None:
                row = insertFrame
            else:
                row = np.hstack([row, getBlankImage(1, targetHeight, colors.White), insertFrame])

        if collage is None:
            collage = row
        else:
            collage = np.vstack([collage, getBlankImage(collage.shape[1], 1, colors.White), row])

    return collage

if __name__ == '__main__':

    imagesPath = glob.glob('images/*/All.JPG')
    for imagePath in imagesPath:
        print(f'\nProcessed: {imagePath}')
        original = loadImage(imagePath)

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        #ret, threshold = cv2.threshold(blur, 10, 255, cv2.THRESH_OTSU)
        ret, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f'found {len(contours)} contours')
        result = original.copy()

        cv2.drawContours(result, contours, -1, colors.Blue, 1)

        zoomRegions = []
        zoomWidth = 25
        zoomHeight = 25

        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            x = int(x)
            y = int(y)


            startX = x - int(zoomWidth / 2)
            endX = x + int(zoomWidth / 2)

            startY = y - int(zoomHeight / 2)
            endY = y + int(zoomHeight / 2)

            if(startX < 0 or endX > original.shape[1]):
                continue

            if (startY < 0 or endY > original.shape[0]):
                continue

            region = original[startY:endY, startX:endX]
            zoomRegions.append(region)

            color = colors.Red
            cv2.circle(result, (x, y), int(radius), color, 1)
            cv2.putText(result, f'{radius:.2f}', (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.Red, 1)
        # cv2.imshow('cropCnt', image)

        collage = create_collages(zoomRegions)
        if(not(collage is None)):
            cv2.imshow('collage', collage)
            bigCollage = getScaledImage(collage, collage.shape[1] * 10)
            cv2.imshow('bigCollage', bigCollage)
            cv2.imwrite(f'{imagePath.replace("All", "Processed")}', bigCollage)

        # for zoomRegion in zoomRegions:
        #     cv2.imshow('zoomRegion', zoomRegion)


        cv2.imshow('gray', gray)
        cv2.imshow('blur', blur)
        cv2.imshow('threshold', threshold)
        cv2.imshow('result', result)
        cv2.waitKey()

    cv2.waitKey()
    cv2.destroyAllWindows()
