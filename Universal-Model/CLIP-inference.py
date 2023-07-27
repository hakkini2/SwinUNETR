import os
import tempfile
import torch
import matplotlib.pyplot as plt

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
)

# SETUP DATA DIRECTORY
os.environ["MONAI_DATA_DIRECTORY"] = "/u/08/hakkini2/unix/ComputerVision/CLIP-and-SwinUNETR/CLIP-Driven-Universal-Model/"
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# TRANSFORMS (???)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = Compose(
    [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        EnsureTyped(keys=["image"], device=device, track_meta=True)
    ]
)


# LOAD TEST DATA (try with task 1 brain tumour data first)
data_dir = "/u/08/hakkini2/unix/ComputerVision/CLIP-and-SwinUNETR/Swin-UNETR-with-MSD/data/Task01_BrainTumour/"
split_json = "dataset.json"

datasets = data_dir + split_json
datalist = load_decathlon_datalist(datasets, True, "test")
test_ds = CacheDataset(
    data=datalist,
    transform=test_transforms,
    cache_num=6,
    cache_rate=1.0,
    num_workers=4
)
test_loader = ThreadDataLoader(test_ds, num_workers=0, batch_size=1)

# Look at an example image
example_img_index = 0
example_img_slice = 90

img_name = os.path.split(test_ds[example_img_index]["image"].meta["filename_or_obj"])[1]
img = test_ds[example_img_index]["image"]
img_shape = img.shape
print("image name: ", img_name, " image shape: ", img_shape)

plt.figure("example image", (18, 6))
title = "Example image " + img_name + ", slice: " + str(example_img_slice)
plt.title(title)
plt.imshow(img[0, :, :, example_img_slice].detach().cpu(), cmap='gray')
plt.savefig('example_image.png')
#plt.show()



#-----------------------------------------------
# Load pre-trained model

#torch.load("/u/08/hakkini2/unix/ComputerVision/CLIP/CLIP-Driven-Universal-Model/swinunetr.pth")
