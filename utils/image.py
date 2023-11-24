import numpy as np


def images_to_grid(image_batch: np.ndarray, line_size: int = 10) -> np.ndarray:
    batch_size, height, width, channels = image_batch.shape
    num_rows = int(np.ceil(batch_size / line_size))
    grid = np.zeros((num_rows * height, line_size * width, channels), dtype=image_batch.dtype)

    for i in range(batch_size):
        row = i // line_size
        col = i % line_size
        grid[row * height:(row + 1) * height, col * width:(col + 1) * width, :] = image_batch[i]

    return grid
