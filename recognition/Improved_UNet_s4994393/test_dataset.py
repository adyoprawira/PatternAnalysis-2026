
from dataset import HipMRIDataset

root = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

for split in ["train", "validate", "test"]:
    dataset = HipMRIDataset(root, split=split)
    image, mask = dataset[0]

    print(f"\n{split.upper()}")
    print("Samples:", len(dataset))
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)
    print("Mask labels:", mask.unique())
