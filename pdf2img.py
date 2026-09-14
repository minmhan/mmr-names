from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import os


def convert(path):
    ''' Convert from PDF directory'''
    for pdf_file in os.listdir(path):
        if not pdf_file.startswith("YG-"):
            continue
        if pdf_file.endswith('.pdf'):
            print("converting -> {}".format(pdf_file))
            pages = convert_from_path("{}/{}".format(path, pdf_file), 500)
            pdf_file = pdf_file[:-4]
            for page in pages:
                page.save("data/img/2019/{}-page-{}.jpg".format(pdf_file, pages.index((page))), "JPEG")

convert('data/2019')