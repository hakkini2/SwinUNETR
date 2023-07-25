import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import torch

import monai
from monai.networks.nets import SwinUNETR

# SETUP DATA DIRECTORY
os.environ["MONAI_DATA_DIRECTORY"] = "/u/08/hakkini2/unix/ComputerVision/SwinUNETR/tutorial-BTCV/"
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


# LOAD TEST DATA
data_dir = "data/"
split_json = "dataset_0.json"

datasets = data_dir + split_json
test_files = monai.data.load_decathlon_datalist(datasets, True, "test")
#test_ds = CacheDataset(data=test_files, )
#test_loader = ThreadDataLoader(test_ds, num_workers=0, batch_size=1)


# LOAD TRAINED MODEL
model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=14,
    feature_size=48,
    use_checkpoint=True,
)
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

# INFERENCE
model.eval()
with torch.no_grad():
    #tests on test data
    # need to do some kind of transforms for the data first 
    pass