import torchvision.transforms as transforms


class ToRGB:
    def __call__(self, img):
        if img.mode == "L":
            return transforms.Grayscale(num_output_channels=3)(img)
        elif img.mode == "RGBA":
            return img.convert("RGB")
        elif img.mode == "CMYK":
            return img.convert("RGB")
        return img