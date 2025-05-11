from paddleocr import PaddleOCR
import os

ocr = PaddleOCR(use_angle_cls=True, lang='ru')
result = ocr.ocr('fsk.jpg', cls=True)
print(result)
with open("ocr_output.txt", "w", encoding="utf-8") as f:
    if result and result[0]:
        for line in result[0]:
            text = line[1][0]
            f.write(text + "\n")
    else:
        print("No text detected.")
