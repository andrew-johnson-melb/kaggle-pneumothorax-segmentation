import cv2
from torch.utils.data import Dataset

class PneumothoraxDataset(Dataset):
    """Dataset FOR Pnemothorax dataset"""

    def __init__(self, files_df, transform=None, labelled=False):
        self.transform = transform
        self.labelled = labelled
        self.files_df = files_df
        
    def __len__(self):
        return len(self.files_df)

    def __getitem__(self, idx):
    
        image = cv2.imread(self.files_df.images.iloc[idx], cv2.IMREAD_COLOR)
        if self.labelled:
            # Make sure to read mask as grey scale, as values are 0, 1
            mask = cv2.imread(self.files_df.masks.iloc[idx], cv2.IMREAD_GRAYSCALE)
            # Binarize mask. 1 for Pneumothorax region and 0 for backgroud
            mask = (mask > 0).astype(float)
            
        if self.transform:
            if self.labelled:
                data = self.transform(image=image,mask=mask)
            else:
                data = self.transform(image=image)
        if self.labelled:
            return data["image"], data["mask"]
        else:
            return data["image"]