import skimage.data
import skimage.transform
import skimage.util

import instance_occlsegm_lib


def main():
    imgs = []
    for img in ['astronaut', 'camera', 'coffee', 'horse']:
        img = getattr(skimage.data, img)()
        imgs.append(img)

    tiled = instance_occlsegm_lib.image.tile(imgs, shape=(2, 2))
    instance_occlsegm_lib.io.imshow(tiled)
    instance_occlsegm_lib.io.waitkey()

    imgs = [skimage.util.img_as_ubyte(skimage.transform.resize(im, (300, 300)))
            for im in imgs]
    tiled = instance_occlsegm_lib.image.tile(imgs, shape=(2, 2))
    instance_occlsegm_lib.io.imshow(tiled)
    instance_occlsegm_lib.io.waitkey()


if __name__ == '__main__':
    main()
