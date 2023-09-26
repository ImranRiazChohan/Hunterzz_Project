from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from pathlib import Path
from PIL import Image

app = FastAPI()

@app.post("/generate_image/")
async def generate_image(
    dear_image: UploadFile = File(...),
    horn_image: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
    horn_size_x: int = Form(...),
    horn_size_y: int = Form(...),
):
    # Save uploaded files locally
    dear_image_path = Path(f"./images/{dear_image.filename}")
    horn_image_path = Path(f"./images/{horn_image.filename}")

    with dear_image_path.open("wb") as dear_image_file:
        dear_image_file.write(dear_image.file.read())

    with horn_image_path.open("wb") as horn_image_file:
        horn_image_file.write(horn_image.file.read())

    # Open Horn Image
    frontImage = Image.open(horn_image_path)

    # Open Background Image
    background = Image.open(dear_image_path)

    # Convert image to RGBA
    frontImage = frontImage.convert("RGBA")

    # Resize horn image
    frontImage = frontImage.resize((horn_size_x, horn_size_y))

    # Convert image to RGBA
    background = background.convert("RGBA")

    # Paste the frontImage at (width, height) where you want to set the horn
    background.paste(frontImage, (width, height), frontImage)

    # Save the generated image
    generated_image_path = Path("./images/final.png")
    background.save(generated_image_path, format="png")

    return FileResponse(generated_image_path, headers={"Content-Disposition": "attachment; filename=new.png"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)