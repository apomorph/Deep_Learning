from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        image_name = self.img_path[index]
        image_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        image = Image.open(image_item_path)
        label = self.label_dir 
        return image, label

    def __len__(self):
        return len(self.img_path)    


root_dir = "data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)       
bees_dataset = MyData(root_dir, bees_label_dir)

# img0, ants_label = ants_dataset[0]
# img0.show()
# img1, bees_label = bees_dataset[0]
# img1.show()

tran_dataset = ants_dataset + bees_dataset