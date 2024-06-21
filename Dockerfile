FROM python:3.9-slim-buster

RUN apt-get update && \
    apt-get -qq -y install tesseract-ocr && \
    apt-get -qq -y install libtesseract-dev

WORKDIR /app/tessdata

COPY requirements.txt requirements.txt

# Copy the downloaded 'ara_numbers' data filedocker build -t my-app-image .
COPY ara_number.traineddata /usr/share/tesseract-ocr/tessdata/


# Install Tesseract-OCR libraries
RUN apt-get install -y tesseract-ocr libtesseract-dev

# Set environment variable for Tesseract data path (adjust path if needed)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata

RUN apt-get update && apt-get install -y libgl1


RUN pip install -r requirements.txt

COPY . .

CMD ["python", "ID_version2.py"]