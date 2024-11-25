from io import StringIO, BytesIO

from django.core.files.uploadedfile import InMemoryUploadedFile

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
import pandas as pd

def pdf_to_text(file):
    def convert(file_):
        pagenums = set()

        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)

        infile = file_.open()

        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
        converter.close()
        text = output.getvalue()
        output.close()
        return text

    def change_file_name_to_txt(filename):
        return ''.join(filename.split(".")[:-1]) + '.txt'

    buf = BytesIO()
    buf.write(convert(file).encode('utf-8'))
    file = InMemoryUploadedFile(buf, 'text/plain', change_file_name_to_txt(file.name), 'text/plain', buf.tell(), 'utf-8')
    return file

def excel_to_text(file):
    def convert(file_):
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_)

        # Extract the 'tweets' column as a list and convert to string
        values = df['values'].tolist()
        text = str(values)

        return text

    def change_file_name_to_txt(filename):
        return ''.join(filename.split(".")[:-1]) + '.txt'

    # Convert the Excel file to text
    text_content = convert(file)

    # Create an in-memory byte buffer
    buf = BytesIO()
    buf.write(text_content.encode('utf-8'))

    # Create an InMemoryUploadedFile instance
    file_name = change_file_name_to_txt(file.name)
    in_memory_file = InMemoryUploadedFile(
        buf,
        None,
        file_name,
        'text/plain',
        buf.tell(),
        'utf-8'
    )

    return in_memory_file