import imageio
import numpy as np

from typing import List

def save_gif(images: List[np.ndarray], path: str):
    shapes = set([image.shape for image in images])
    assert len(shapes) == 1, 'Images have different shape: {shapes}'
    h, w, c = list(shapes)[0]
    assert c in [1, 3], f'Unexpected channels size. Shape: [{h}, {w}, {c}]'

    image_list = [np.squeeze(image).astype(np.uint8) for image in images]
    imageio.mimwrite(
        path,
        image_list,
        fps=5,
    )
