#pip install tensorboard
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
img_path = "data\\train\\ants\\0013035.jpg"
img = Image.open(img_path)
img_array = np.array(img)

writer.add_image("test", img_array, 1, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()

#tensorboard --logdir=logs
