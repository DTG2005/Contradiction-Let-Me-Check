from Main_Operations.ocr3 import ocr_func
import os

def read_img_file(file_path):
        try:
            file_path = os.path.abspath('{}'.format(file_path))
            text = ocr_func(file_path)
            return text
        except Exception as e:
            return f"An error occurred reading DOC file: {e}"
        