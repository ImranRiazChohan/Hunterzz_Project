import cv2
import matplotlib.pyplot as plt
from PIL import Image

def Final_Image(dear_image,horn_image,width,height,horn_size_x,horn_size_y):    
    # Open Horn Image
    frontImage = Image.open(horn_image)
  
    # Open Background Image
    background = Image.open(dear_image)
  
    # Convert image to RGBA
    frontImage = frontImage.convert("RGBA")

    #resize  horn image 
    frontImage=frontImage.resize((horn_size_x,horn_size_y))
    
    # Convert image to RGBA
    background = background.convert("RGBA")

    # Paste the frontImage at (width, height) jahan horn set krna hai 
    background.paste(frontImage, (width, height), frontImage)

    return background.save("./images/new.png", format="png"),'Successfully Created!'

if __name__ == "__main__":
    result,message=Final_Image('./images/dear.jpg','./images/output.png',200,30,200,100)
    print(message)
    output_image=Image.open('./images/new.png')
    plt.imshow(output_image)