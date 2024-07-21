import fitz  # PyMuPDF
from docx import Document
from odf.opendocument import load as load_odt
from odf.text import P
import os
from .ocr3 import ocr_func

import pypandoc

def docread(path: str):
    def read_text_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            return f"An error occurred reading text file: {e}"
        
    def read_pdf_file_alternate(file_path):
        try:
            document = fitz.open(file_path)
            text= ""
            for page in document: 
                pix = page.get_pixmap(matrix=fitz.Identity, dpi=None, 
                          colorspace=fitz.csRGB, clip=None, alpha=True, annots=True) 
                pix.save("samplepdfimage-%i.jpg" % page.number)  # save file 
                text+= ocr_func("samplepdfimage-%i.jpg" % page.number)
                os.remove("samplepdfimage-%i.jpg" % page.number)
            return text
        except Exception as e:
            return f"An error occurred reading PDF file: {e}"

    def read_pdf_file(file_path):
        try:
            document = fitz.open(file_path)
            text = ""
            for page_num in range(len(document)):
                page = document.load_page(page_num)
                text += page.get_text()
            return text
        except Exception as e:
            return f"An error occurred reading PDF file: {e}"

    def read_docx_file(file_path):
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"An error occurred reading DOCX file: {e}"

    def read_odt_file(file_path):
        try:
            doc = load_odt(file_path)
            paragraphs = doc.getElementsByType(P)
            text = ""
            for paragraph in paragraphs:
                for content in paragraph.childNodes:
                    if content.nodeType == 3:  # Node.TEXT_NODE
                        text += content.data
                text += "\n"
            return text
        except Exception as e:
            return f"An error occurred reading ODT file: {e}"

    def read_doc_file(file_path):
        try:
            text = pypandoc.convert_file(file_path, 'plain')
            return text
        except Exception as e:
            return f"An error occurred reading DOC file: {e}"
        
    def read_img_file(file_path):
        try:
            file_path = os.path.abspath('{}'.format(file_path))
            text = ocr_func(file_path)
            return text
        except Exception as e:
            return f"An error occurred reading DOC file: {e}"

    def read_document(file_path):
        if file_path.endswith('.txt'):
            return read_text_file(file_path)
        elif file_path.endswith('.pdf'):
            return read_pdf_file(file_path)
        elif file_path.endswith('.docx'):
            return read_docx_file(file_path)
        elif file_path.endswith('.odt'):
            return read_odt_file(file_path)
        elif file_path.endswith('.doc'):
            return read_doc_file(file_path)
        elif file_path.endswith('.png'):
            return read_img_file(file_path)
        elif file_path.endswith('.jpg'):
            return read_img_file(file_path)
        elif file_path.endswith('.jpeg'):
            return read_img_file(file_path)
        else:
            return "Unsupported file format. Please provide a .txt, .pdf, .docx, .odt, or .doc file."
        
    return read_document(path)