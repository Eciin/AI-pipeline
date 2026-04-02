import cv2 as cv
import sys
from pathlib import Path

img_path = next(Path("./input").iterdir())
img = cv.imread(str(img_path))
if img is None:
    sys.exit("Could not read the image.")

# 1) grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2) upscale
gray = cv.resize(gray, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

# 3) denoise a bit
gray = cv.GaussianBlur(gray, (3, 3), 0)

# 4) threshold
processed = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


cv.imwrite("./output/Opencv/processed_for_ocr.png", processed)