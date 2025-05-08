import easyocr

reader = easyocr.Reader(['en', 'ru'])
result = reader.readtext('fsk.jpg', detail=0, paragraph=True)
with open("ocr_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(result))
