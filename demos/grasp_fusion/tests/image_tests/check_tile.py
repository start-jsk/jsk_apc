import skimage.data
import skimage.transform
import skimage.util

import grasp_fusion_lib


def main():
    imgs = []
    for img in ['astronaut', 'camera', 'coffee', 'horse']:
        img = getattr(skimage.data, img)()
        imgs.append(img)

    tiled = grasp_fusion_lib.image.tile(imgs, shape=(2, 2))
    grasp_fusion_lib.io.imshow(tiled)
    grasp_fusion_lib.io.waitkey()

    imgs = [skimage.util.img_as_ubyte(skimage.transform.resize(im, (300, 300)))
            for im in imgs]
    tiled = grasp_fusion_lib.image.tile(imgs, shape=(2, 2))
    grasp_fusion_lib.io.imshow(tiled)
    grasp_fusion_lib.io.waitkey()


if __name__ == '__main__':
    main()
