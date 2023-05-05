FROM python:3.8
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
ENV NAME World
CMD ["python", "i191762_i191694.py"]
