import os
from torch.utils.data import Dataset
from PIL import Image

class BirdDataset(Dataset):
    def __init__(self, image_paths: str, image_dir: str, segmentation_dir: str, transform_image, transform_mask):
        super().__init__()
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.samples = []
        
        with open(image_paths, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # e.g., "1 001.Black_Footed_Albatross/Black_Footed_Albatross_0001_30.jpg"
            rel_path = line.split(" ")[-1].strip()
            
            # UPGRADE: Handles different operating system slashes smoothly
            clean_rel_path = rel_path.replace('\\', '/')
            
            base_name = os.path.splitext(clean_rel_path)[0]
            
            # Maps perfectly to: data/images/001.Black_Footed_Albatross/...jpg
            img_path = os.path.join(image_dir, f"{base_name}.jpg")
            mask_path = os.path.join(segmentation_dir, f"{base_name}.png")
            
            # Double check files exist before appending
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]
        
        image = Image.open(img_path).convert("RGB")
        seg = Image.open(mask_path).convert("L")
        
        image = self.transform_image(image)
        seg = self.transform_mask(seg)
        return image, seg