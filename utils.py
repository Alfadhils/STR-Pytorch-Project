import numpy as np
import easyocr
import cv2

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return [], []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    pick_probs = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    if probs is not None:
        idxs = probs

    idxs = np.argsort(idxs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        pick_probs.append(probs[i] if probs is not None else None)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int"), pick_probs

def decode_predictions(scores, geometry, min_confidence=0.5):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	for y in range(0, numRows):
		scoresData = scores[0, 0, y]

		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]

		anglesData = geometry[0, 4, y]

		for x in range(0, numCols):
			if scoresData[x] < min_confidence:
				continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
			
	return (rects, confidences)

def recognize(img):
    reader = easyocr.Reader(['en'])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ocr_result = reader.readtext(gray)
    if ocr_result :
        text = ocr_result[0][1]
    else :
        text = '-'
    
    return text

def map_label(char_array):
    result = []
    for char in char_array:
        if char.isdigit():
            result.append(int(char))
        elif char.isalpha():
            if char.isupper():
                result.append(ord(char) - ord('A') + 10)
            elif char.islower():
                result.append(ord(char) - ord('a') + 36)
        else:
            result.append(None)
    return np.array(result, dtype=np.int64)

def map_result(result_array):
    char = []
    for num in result_array:
        if 0 <= num <= 9:
            char.append(str(num))
        elif 10 <= num <= 35:
            char.append(chr(ord('A') + num - 10))
        elif num is not None:
            char.append(chr(ord('a') + num - 36))
        else:
            char.append(None)

    return ''.join([str(c) for c in char if c is not None])