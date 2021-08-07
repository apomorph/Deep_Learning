from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import cv2  #pip install opencv-python

img_path = "data\\train\\ants\\0013035.jpg"
img = Image.open(img_path)

trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)

# print(img_tensor)

writer = SummaryWriter("logs")
writer.add_image("tensor", img_tensor)

img_cv = cv2.imread("data\\train\\bees\\16838648_415acd9e3f.jpg")
writer.add_image("tensor", img_cv, 2, dataformats='HWC')

writer.close()

#tensorboard --logdir=logs

#Normalize
img = Image.open("data\\train\\bees\\2405441001_b06c36fa72.jpg")
img_tensor = trans_tensor(img)
writer = SummaryWriter("logs")
writer.add_image("Normalize", img_tensor)

trans_norm = transforms.Normalize([1,2,2],[2,1,1])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm, 1)

#Resize
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)
writer.add_image("Resize", img_resize)

writer.close()