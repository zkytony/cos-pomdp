import os
from PIL import Image

def xyxy_to_xywh(box, size, center=False, normalize=False):
    """
    Converts bounding box format from 'xyxy'
    to 'xywh'.

    Args:
        box: [Upper Left x, Upper Left y, Lower Right x, Lower Right y]
        size: [image width, image height]
        center (bool): If True, then the x, y refer to center coordinate. Otherwise,
                       it will be the top-left.
        normalize (bool): If True, then the output x, y, w, h will be 0-1
    Returns:
        x, y, w, h

    References:
    - https://github.com/ultralytics/yolov3/issues/26
    - https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#2-create-labels
    """
    #[Upper Left x, Upper Left y, Lower Right x, Lower Right y]
    img_width, img_height = size
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    x = x1
    y = y1
    w = box_width
    h = box_height
    if center:
        x = ((x1 + x2) / 2)
        y = ((y1 + y2) / 2)
    if normalize:
        x /= img_width
        y /= img_height
        w /= img_width
        h /= img_width
    return x, y, w, h


def saveimg(img, path):
    """
    img: numpy array of image. RGB.
    """
    im = Image.fromarray(img)
    im.save(os.path.join(path))
