import base64
import io

import numpy as np
import PIL.Image


# binary -> XXXX


def binary_to_base64(img_binary):
    # binary -> base64
    img_base64 = base64.encodestring(img_binary)
    img_base64 = img_base64.decode('utf-8')
    return img_base64


def binary_to_ndarray(img_binary):
    # binary -> base64 -> ndarray
    img_base64 = binary_to_base64(img_binary)
    img_ndarray = base64_to_ndarray(img_base64)
    return img_ndarray


def binary_to_pil(img_binary):
    # binary -> base64 -> pil
    img_base64 = binary_to_base64(img_binary)
    img_pil = base64_to_pil(img_base64)
    return img_pil


# base64 -> XXXX


def base64_to_binary(img_base64):
    # base64 -> binary
    img_binary = base64.decodestring(img_base64)
    return img_binary


def base64_to_pil(img_base64):
    # base64 -> pil
    f = io.BytesIO()
    f.write(base64.b64decode(img_base64))
    img_pil = PIL.Image.open(f)
    return img_pil


def base64_to_ndarray(img_base64):
    # base64 -> pil -> ndarray
    img_pil = base64_to_pil(img_base64)
    img_ndarray = pil_to_ndarray(img_pil)
    return img_ndarray


# ndarray -> XXXX


def ndarray_to_pil(img_ndarray):
    # ndarray -> pil
    img_pil = PIL.Image.fromarray(img_ndarray)
    return img_pil


def ndarray_to_binary(img_ndarray):
    # ndarray -> pil -> binary
    img_pil = ndarray_to_pil(img_ndarray)
    img_binary = pil_to_binary(img_pil)
    return img_binary


def ndarray_to_base64(img_ndarray):
    # ndarray -> pil -> binary -> base64
    img_binary = ndarray_to_binary(img_ndarray)
    img_base64 = binary_to_base64(img_binary)
    return img_base64


# PIL -> XXXX


def pil_to_base64(img_pil):
    # pil -> binary -> base64
    img_binary = pil_to_binary(img_pil)
    img_base64 = binary_to_base64(img_binary)
    return img_base64


def pil_to_binary(img_pil):
    # pil -> binary
    f = io.BytesIO()
    img_pil.save(f, format='JPEG')
    img_binary = f.getvalue()
    return img_binary


def pil_to_ndarray(img_pil):
    # pil -> ndarray
    img_ndarray = np.array(img_pil)
    return img_ndarray


if __name__ == '__main__':
    import skimage.data
    img = skimage.data.coffee()
    img_b64 = ndarray_to_base64(img)
    print(img_b64)
    print(type(img_b64))
