import numpy as np


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
    Returns:
        a float numpy array of shape [4],
            squared bounding boxes.
    """

    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = bboxes
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[2] = square_bboxes[0] + max_side - 1.0
    square_bboxes[3] = square_bboxes[1] + max_side - 1.0
    return square_bboxes


def scale_box(box, scale):
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    new_w, new_h = w * scale, h * scale
    y1, y2, x1, x2 = center_y - new_h / 2, center_y + new_h / 2, center_x - new_w / 2, center_x + new_w / 2,
    return np.array((x1, y1, x2, y2))

def scale_tensor_box(box, scale):
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    new_w, new_h = w * scale, h * scale
    y1, y2, x1, x2 = center_y - new_h / 2, center_y + new_h / 2, center_x - new_w / 2, center_x + new_w / 2,
    return (x1, y1, x2, y2)