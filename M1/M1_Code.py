import numpy as np
import cv2
import matplotlib.pyplot as plt
import supervision as sv
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from rembg import remove
from PIL import Image


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

  #Save Output Image
  cv2.imwrite(f'./images/output_image.jpg',Output_img)

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


if __name__=='__main__':
  box_dict={'x':22,'y':8,'width':498,"height":289}

  box = np.array([
      box_dict['x'],
      box_dict['y'],
      box_dict['x'] + box_dict['width'],
      box_dict['y'] + box_dict['height']
  ])

  print('Apply SAM Model')
  output_img=Segment_anything_using_SAM('./images/antler-antler-carrier-fallow-deer-hirsch.jpg',box)
  print('Now Remove Background')
  Remove_Background('./images/output_image.jpg')