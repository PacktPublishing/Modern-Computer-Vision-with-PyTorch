import os, io
from fmnist import FMNIST
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

model = FMNIST()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/files", StaticFiles(directory="files"), name="files")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post('/uploaddata/')
async def upload_file(request: Request, file:UploadFile=File(...)):
    print(request)
    content = file.file.read()
    saved_filepath = f'files/{file.filename}'
    with open(saved_filepath, 'wb') as f:
        f.write(content)
    output = model.predict_from_path(saved_filepath)
    payload = {'request': request, 
        "filename": file.filename, 
        'output': output}
    return templates.TemplateResponse("home.html", payload)

@app.post("/predict")
def predict(request: Request, file:UploadFile=File(...)):
    content = file.file.read()
    image = Image.open(io.BytesIO(content)).convert('L')
    output = model.predict_from_image(image)
    return output
