import math
from dataclasses import dataclass
from pathlib import Path

from os import path

import numpy as np
import cv2
import glob

import tkinter as tk
import tkinter.ttk as ttk

from tkinter import filedialog

from EniPy import colors
from EniPy import eniUtils

@dataclass
class Circle:
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0

    def intX(self) -> int:
        return int(self.x)
    def intY(self) -> int:
        return int(self.y)
    def intCenter(self) -> (int, int):
        return (int(self.x), int(self.y))
    def intRadius(self) -> int:
        return int(self.radius)
@dataclass
class RegionValue:
    sum: float = 0.0
    min: int = -1
    max: int = -1
    count: int = 0
    def average(self) -> float:
        if self.count != 0:
            return self.sum / self.count
        return 0

@dataclass
class Marker:
    image: None = None
    angleCw: float = 0.0
    radius: float = 0.0
    x: float = 0.0
    y: float = 0.0
    def intX(self) -> int:
        return int(self.x)
    def intY(self) -> int:
        return int(self.y)

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def angle(a, b):
    dot = np.dot(a, b)
    lenA = cv2.norm(a)
    lenB = cv2.norm(b)
    a = math.acos(dot / (lenA * lenB))
    return a
def clockAngle(a, b):
    dot = np.dot(a, b)
    det = a[0] * b[1] - a[1] * b[0]
    a = math.atan2(det, dot)
    if a < 0:
        a = math.pi + (math.pi + a)
    return a
def orderPoints(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect
def getScaledImage(image, targetWidth = 1920):
    scale = targetWidth / image.shape[1]
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    scaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return scaled

def loadImage(filename):
    image = cv2.imread(filename)
    return image

def getBlankImage(width, height, color = colors.Black):
    blankImage = np.zeros((height, width, 3), np.uint8)
    blankImage[:] = color
    return blankImage
def create_collages(images, scale):

    collage_size = int(math.ceil(math.sqrt(len(images))))

    collage = None

    if len(images) > 0:
        targetWidth = images[0].shape[1] * scale
        targetHeight = images[0].shape[0] * scale

    for colIndex in range(collage_size):
        row = None
        for rowIndex in range(collage_size):
            i = colIndex * collage_size + rowIndex
            insertFrame = getBlankImage(targetWidth, targetHeight, colors.White)
            if i < len(images):
                insertFrame = getScaledImage(images[i], images[i].shape[1] * scale)

                gray = cv2.cvtColor(insertFrame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)
                ret, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f'found {len(contours)} contours')
                mean = -1
                min = -1
                max = -1
                if len(contours) > 0:
                    mask = np.zeros(gray.shape, np.uint8)
                    cv2.drawContours(mask, contours[0], -1, 255, -1)
                    regionValue = getRegionValue(gray, mask)
                    min = regionValue.min
                    max = regionValue.max
                    mean = regionValue.average()

                cv2.drawContours(insertFrame, contours, -1, colors.Blue, 1)
                cv2.putText(insertFrame, f'{i} {int(mean)} [{min};{max}]', (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.Red, 1)


            if row is None:
                row = insertFrame
            else:
                row = np.hstack([row, getBlankImage(1, targetHeight, colors.White), insertFrame])

        if collage is None:
            collage = row
        else:
            collage = np.vstack([collage, getBlankImage(collage.shape[1], 1, colors.White), row])

    return collage

def markersCheck(path):
    imagesPath = glob.glob(f'{path}/*.jpg')
    for imagePath in imagesPath:
        print(f'\nProcessed: {imagePath}')
        original = loadImage(imagePath)

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # ret, threshold = cv2.threshold(blur, 10, 255, cv2.THRESH_OTSU)
        ret, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f'found {len(contours)} contours')
        result = original.copy()

        cv2.drawContours(result, contours, -1, colors.Blue, 1)

        rect = cv2.minAreaRect(np.vstack(contours))
        box = cv2.boxPoints(rect)
        box = orderPoints(box)
        centerMid = midpoint(box[0], box[2])
        bottomMid = midpoint(box[2], box[3])
        cv2.drawContours(result, [np.intp(box)], 0, colors.Cyan, 1, lineType=cv2.LINE_AA)

        cv2.circle(result, np.intp(centerMid), 5, colors.Cyan)
        cv2.circle(result, np.intp(bottomMid), 5, colors.Cyan)

        zoomWidth = 25
        zoomHeight = 25
        markerList = []
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            m = Marker()
            m.x = x
            m.y = y
            m.radius = radius

            baseLine = (bottomMid[0] - centerMid[0], bottomMid[1] - centerMid[1])
            currentLine = (x - centerMid[0], y - centerMid[1])

            m.angleCw = clockAngle(baseLine, currentLine)

            startX = m.intX() - int(zoomWidth / 2)
            endX = m.intX() + int(zoomWidth / 2)

            startY = m.intY() - int(zoomHeight / 2)
            endY = m.intY() + int(zoomHeight / 2)

            if (startX < 0 or endX > original.shape[1]):
                continue

            if (startY < 0 or endY > original.shape[0]):
                continue

            m.image = original[startY:endY, startX:endX]

            markerList.append(m)

        markerList.sort(key=lambda x: x.angleCw)

        markesImages = []
        for i, marker in enumerate(markerList):
            print(f'{i} {marker.radius}')
            cv2.circle(result, (marker.intX(), marker.intY()), int(radius), colors.Red, 1)
            cv2.putText(result, f'{i}:{radius:.2f}', (marker.intX(), marker.intY() + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.Red, 1)
            markesImages.append(marker.image)


        collage = create_collages(markesImages, 10)
        if (not (collage is None)):
            #cv2.imshow('collage', collage)
            bigCollage = getScaledImage(collage, collage.shape[1] * 1)
            cv2.imshow('bigCollage', bigCollage)
            #cv2.imwrite(f'{imagePath.replace(".jpg", ".Processed.jpg")}', bigCollage)

        #cv2.imshow('gray', gray)
        #cv2.imshow('blur', blur)
        #cv2.imshow('threshold', threshold)
        cv2.imshow('result', result)
        cv2.waitKey()

    cv2.destroyAllWindows()
def getMaxCircle(contours):
    result = Circle()

    for i in range(len(contours)):
        print(f'Edges {len(contours[i])}')
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        if (radius > result.radius):
            result = Circle(x, y, radius)

    return result
def getRegionValue(image, mask):
    result = RegionValue()

    actualRegion = cv2.bitwise_and(image, mask)
    r = np.where(actualRegion > 0)
    subRegion = actualRegion[r]
    result.sum = np.sum(subRegion)

    if subRegion.size > 0:
        result.min = np.min(subRegion)
        result.max = np.max(subRegion)
    result.count = np.count_nonzero(mask)
    return result
def findAverageCircularIntensity(targetImageDescriptions, resultFilename = 'result.txt'):
    output = open(resultFilename, "w")

    for imageDescription in targetImageDescriptions:
        print(f'\nProcessed: {imageDescription.path}')
        output.write(f'{imageDescription.value()}\t')
        image = loadImage(str(imageDescription.path))
        if image is None:
            print(f'image corrupted')
            continue
        imageScale = 2
        original = getScaledImage(image, 640 * imageScale)
        result = original.copy()
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (15, 15), 5)
        ret, threshold = cv2.threshold(blur,32,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f'found {len(contours)} contours')
        cv2.imshow('threshold', threshold)
        width = result.shape[1]
        height = result.shape[0]

        centerCircle = getMaxCircle(contours)

        if centerCircle.radius > 0:
            print(f'center {centerCircle.intX() / imageScale} {centerCircle.intY() / imageScale}')
            step = 50 * imageScale
            prevRadius = 0
            for radius in [10 * imageScale] + list(range(50 * imageScale, min(int(width / 2), centerCircle.intRadius() - step), step)) + [centerCircle.intRadius()]:
                mask = np.zeros_like(gray)
                cv2.circle(mask, centerCircle.intCenter(), radius, colors.White, -1)
                if prevRadius != 0:
                    cv2.circle(mask, centerCircle.intCenter(), prevRadius, colors.Black, -1)
                prevRadius = radius

                regionValue = getRegionValue(gray, mask)

                print(f'Count: {regionValue.count} Sum1: {regionValue.sum} Avr: {regionValue.average()}')
                output.write(f'{regionValue.average()}\t')

                cv2.circle(result, centerCircle.intCenter(), radius, colors.Red)
                cv2.putText(result, f'{int(radius / imageScale)}', (centerCircle.intX() + radius, centerCircle.intY()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.Magenta, 1)
                cv2.putText(result, f'{regionValue.min}:{regionValue.max}', (centerCircle.intX() + radius, centerCircle.intY() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.Magenta, 1)
                cv2.putText(result, f'{int(regionValue.average())}', (centerCircle.intX() + radius, centerCircle.intY() + 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.Magenta, 1)
        else:
            print(f'No circle found')



        cv2.putText(result, f'{imageDescription.value()}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, colors.Red, 2)
        cv2.imshow('result', result)
        cv2.imwrite(f'{imageDescription.value()}.processed.png', result)
        mark = cv2.waitKey() - ord('0')
        output.write(f'{centerCircle.radius}\t')
        output.write(f'{mark}\t')
        output.write(f'\n')
        print(f'mark = {mark}')

    cv2.destroyAllWindows()
@dataclass
class TargetImageDescription:
    path: Path
    def value(self) -> int:
        return int(self.path.stem)
    def __lt__(self, other):
        return self.value() < other.value()
def onStartButtonClick():
    print(f'onClick {fromIndex.get()} {endIndex.get()}')
    targets = []
    for description in availableDescriptions:
        if description.value() >= fromIndex.get() and description.value() < endIndex.get():
            targets.append(description)
    print(f'in selected range {len(targets)} values')
    availableImagesCountLabel["text"] += f':{len(targets)}'
    findAverageCircularIntensity(targets, f'{fromIndex.get()}_{endIndex.get()}.txt')

def onStartButtonClickMarkers():
    print(f'onClick')
    markersCheck(pathLabelMarkers["text"])

availableDescriptions = []
def onPathSelectionClick():
    global availableDescriptions
    print(f'onPathSelectionClick')
    path = tk.filedialog.askdirectory()
    pathLabel["text"] = path
    availableDescriptions = []
    for name in glob.glob(f'{path}/*.png'):
        availableDescriptions.append(TargetImageDescription(Path(name)))

    availableDescriptions.sort()
    availableImagesCountLabel["text"] = f'{len(availableDescriptions)}'

def onPathSelectionClickMarkers():
    print(f'onPathSelectionClick')
    path = tk.filedialog.askdirectory()
    pathLabelMarkers["text"] = path

root = tk.Tk()
root.title("IrLedsChecker")
root.geometry("320x240")

tabControl = ttk.Notebook(root)
circularIntensityTab = ttk.Frame(tabControl)
markersCheckTab = ttk.Frame(tabControl)

tabControl.add(circularIntensityTab, text='CircularIntensity')
tabControl.add(markersCheckTab, text='MarkersCheck')

tabControl.pack(expand=1, fill="both")

pathLabel = ttk.Label(circularIntensityTab)
pathLabel.pack(fill=tk.X)

pathSelectionButton = ttk.Button(circularIntensityTab, text="...", command = onPathSelectionClick)
pathSelectionButton.pack(anchor=tk.N, fill=tk.X)


availableImagesCountLabel = ttk.Label(circularIntensityTab)
availableImagesCountLabel.pack(anchor=tk.N, fill=tk.X)

fromIndex = tk.IntVar()
endIndex = tk.IntVar()

fromEntry = ttk.Entry(master=circularIntensityTab, textvariable=fromIndex)
fromEntry.pack(anchor=tk.N, side = tk.LEFT, padx=6)

toEntry = ttk.Entry(master=circularIntensityTab, textvariable=endIndex)
toEntry.pack(anchor=tk.N, side = tk.RIGHT, padx=6)

startButton = ttk.Button(circularIntensityTab, text="Start", command=onStartButtonClick)
startButton.pack(anchor=tk.N)


pathLabelMarkers = ttk.Label(markersCheckTab, text=path.expandvars(r'%USERPROFILE%/Desktop/IoIrLeds/2024.01.10AfterNight'))
pathLabelMarkers.pack(fill=tk.X)

pathSelectionButtonMarkers = ttk.Button(markersCheckTab, text="...", command = onPathSelectionClickMarkers)
pathSelectionButtonMarkers.pack(anchor=tk.N, fill=tk.X)

startButtonMarkers = ttk.Button(markersCheckTab, text="Start", command=onStartButtonClickMarkers)
startButtonMarkers.pack(anchor=tk.N)

if __name__ == '__main__':
    root.mainloop()
