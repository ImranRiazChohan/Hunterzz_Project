from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import os
import numpy as np
import cv2
from PIL import Image
from rembg import remove
import supervision as sv
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pathlib import Path
from fastapi.param_functions import Depends
from fastapi.responses import JSONResponse

# [{'x': 22, 'y': 8, 'width': 498, 'height': 289, 'label': ''}

app = FastAPI()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH='sam_vit_h_4b8939.pth'
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device='cpu')
mask_predictor = SamPredictor(sam)


def Segment_anything_using_SAM(IMAGE_PATH,box):

  image_bgr = cv2.imread(IMAGE_PATH)
  image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  
  mask_predictor.set_image(image_rgb)

  masks, scores, logits = mask_predictor.predict(box=box,multimask_output=False)
  box_annotator = sv.BoxAnnotator(color=sv.Color.red())
  mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

  detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks),mask=masks)
  detections = detections[detections.area == np.max(detections.area)]

  source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
  segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

  source_image_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
  segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

  segmentation_mask = masks[0]
  image=source_image
  binary_mask = np.where(segmentation_mask > 0.5, 1, 0)

  # Create an RGBA image with transparent background
  height, width, _ = image.shape
  Output_img = np.zeros((height, width, 4), dtype=np.uint8)

  # Set the RGB values for the masked region
  Output_img[..., :3] = image * binary_mask[..., np.newaxis]

  # Set the alpha channel to 255 (fully opaque) for the masked region and 0 (fully transparent) outside the masked region
  Output_img[..., 3] = (binary_mask * 255).astype(np.uint8)


  return Output_img


def Remove_Background(image_path):
  # Store path of the output image in the variable output_path
  output_path = './images/final_image.png'
  # Processing the image
  input = Image.open(image_path)
  # Removing the background from the given Image
  output = remove(input)
  #Saving the image in the given path
  output.save(output_path)
  return 'Succefully RemoveBackground'

# Define FastAPI endpoints

@app.get("/show_horn_image")
async def main():
    img_path = Path('./images/horns.png')
    return FileResponse(img_path, media_type="image/png", filename="horns.png")


@app.post("/segment/")
async def segment_image(image: UploadFile,x:int,y:int,width:int,height:int):
    image_path = "./images/temp_image.jpg"  # Temporary path to save the uploaded image
    with open(image_path, "wb") as f:
        f.write(image.file.read())

    box = np.array([x,y,x+width,y+height])
    print('box:',box)
    # Perform segmentation using SAM
    output_image = Segment_anything_using_SAM(image_path, box)

    # Save the segmented image
    segmented_output_path = './images/horns.png'
    cv2.imwrite(segmented_output_path, output_image)

    #Remove Background of Image
    bg_remove_img=Remove_Background(segmented_output_path)
    # Clean up temporary image
    os.remove(image_path)

    return {"message": "Segmentation complete", "segmented_download_link": segmented_output_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)