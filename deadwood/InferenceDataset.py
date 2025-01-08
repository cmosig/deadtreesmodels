import rasterio
from rasterio import windows
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class InferenceDataset(Dataset):

    def __init__(self, image_path, tile_size=512, padding=56):
        super(InferenceDataset, self).__init__()
        self.image_path = image_path
        self.tile_size = tile_size
        self.padding = padding
        self.image_src = rasterio.open(self.image_path)
        self.width = self.image_src.width
        self.height = self.image_src.height

        self.cropped_windows = [
            window for window in get_windows(
                xmin=self.padding,
                ymin=self.padding,
                xmax=self.width - self.padding,
                ymax=self.height - self.padding,
                tile_width=self.tile_size - (padding * 2),
                tile_height=self.tile_size - (padding * 2),
                overlap=0,
            )
        ]

    def __len__(self):
        return len(self.cropped_windows)

    def __getitem__(self, idx):
        image_src = rasterio.open(self.image_path)
        cropped_window = self.cropped_windows[idx]
        cropped_window_dict = {
            "col_off": cropped_window.col_off,
            "row_off": cropped_window.row_off,
            "width": cropped_window.width,
            "height": cropped_window.height,
        }
        inference_window = windows.Window(
            cropped_window.col_off - self.padding,
            cropped_window.row_off - self.padding,
            cropped_window.width + (2 * self.padding),
            cropped_window.height + (2 * self.padding),
        )
        image = image_src.read((1, 2, 3), window=inference_window)

        # Reshape the image tensor to have 3 channels
        image = image.transpose(1, 2, 0)

        image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        image_tensor = image_transforms(image).float().contiguous()
        return image_tensor, cropped_window_dict


def get_windows(xmin, ymin, xmax, ymax, tile_width, tile_height, overlap):
    xstep = tile_width - overlap
    ystep = tile_height - overlap
    for x in range(xmin, xmax, xstep):
        if x + tile_width > xmax:
            x = xmax - tile_width
        for y in range(ymin, ymax, ystep):
            if y + tile_height > ymax:
                y = ymax - tile_height
            window = windows.Window(x, y, tile_width, tile_height)
            yield window
