FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./installs /app/installs
# Below three lines are done because torch is a very large file, and we want to avoid downloading it
# again and again
RUN pip install ./installs/numpy-1.19.0-cp37-cp37m-manylinux2010_x86_64.whl
RUN pip install ./installs/future-0.18.2.tar.gz
RUN pip install ./installs/torch-1.5.1-cp37-cp37m-manylinux1_x86_64.whl
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY . /app
RUN ls /app
RUN rm -rf /app/installs
EXPOSE 5000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]
