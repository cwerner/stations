# as defined in PyTorch, custom extension
import pathlib
from typing import BinaryIO, List, Optional, Text, Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid


def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format: Optional[str] = None,
    label: Optional[str] = None,
    label2: Optional[str] = None,
) -> Image:

    grid = make_grid(
        tensor,
        nrow=nrow,
        padding=padding,
        pad_value=pad_value,
        normalize=normalize,
        range=range,
        scale_each=scale_each,
    )
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    if label:
        d = ImageDraw.Draw(im)
        fnt = ImageFont.load_default()
        x, _ = im.size
        y = 5

        # iteration label
        w, h = fnt.getsize(label)
        d.rectangle((x - w - 4, y, x - 2, y + h), fill="black")
        d.text((x - w - 2, y), label, fnt=fnt, fill=(255, 255, 0))

        if label2:
            # model label
            w, h = fnt.getsize(label2)
            d.rectangle((2, y, w + 4, y + h), fill="black")
            d.text((4, y), label2, fnt=fnt, fill=(255, 255, 0))

    im.save(fp, format=format)
    return im
