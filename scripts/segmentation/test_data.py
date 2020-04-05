import os
import time


def _get_mhp_pairs_v1(folder, split='train'):
    img_paths = []
    mask_paths = []

    if split == 'train':
        img_list = os.path.join(folder, 'list', 'train.txt')
        img_folder = os.path.join(folder, 'train', 'images')
        mask_folder = os.path.join(folder, 'train', 'parsing_annos')
    elif split == 'val':
        img_list = os.path.join(folder, 'list', 'val.txt')
        img_folder = os.path.join(folder, 'val', 'images')
        mask_folder = os.path.join(folder, 'val', 'parsing_annos')
    else:
        raise (RuntimeError("Unsupported split mode : " + split + "\n"))

    with open(img_list) as txt:
        for basename in txt:
            # record mask paths
            mask_short_path = []
            basename = basename.rstrip('\n')
            for maskname in os.listdir(mask_folder):
                if maskname.startswith(basename):
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(maskpath):
                        mask_short_path.append(maskpath)
                    else:
                        print('cannot find the mask:', maskpath)

            # mask_short_path is not empty
            if mask_short_path:
                mask_paths.append(mask_short_path)

            # record img paths
            imgname = basename + '.jpg'
            imgpath = os.path.join(img_folder, imgname)
            if os.path.isfile(imgpath):
                img_paths.append(imgpath)
            else:
                print('cannot find the image:', imgpath)

    return img_paths, mask_paths


def _get_mhp_pairs_v2(folder, split='train'):
    img_paths = []
    mask_paths = []

    if split == 'train':
        img_list = os.path.join(folder, 'list', 'train.txt')
        img_folder = os.path.join(folder, 'train', 'images')
        mask_folder = os.path.join(folder, 'train', 'parsing_annos')
    elif split == 'val':
        img_list = os.path.join(folder, 'list', 'val.txt')
        img_folder = os.path.join(folder, 'val', 'images')
        mask_folder = os.path.join(folder, 'val', 'parsing_annos')
    else:
        raise (RuntimeError("Unsupported split mode : " + split + "\n"))

    mask_list = os.listdir(mask_folder)

    with open(img_list) as txt:
        for basename in txt:
            # record mask paths
            mask_short_path = []
            basename = basename.rstrip('\n')

            for maskname in mask_list:
                if maskname.startswith(basename):
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(maskpath):
                        mask_short_path.append(maskpath)
                    else:
                        print('cannot find the mask:', maskpath)

            # mask_short_path is not empty
            if mask_short_path:
                mask_paths.append(mask_short_path)

            # remove the added element to speed up searching
            mask_list = list(set(mask_list).difference(set(mask_short_path)))

            # record img paths
            imgname = basename + '.jpg'
            imgpath = os.path.join(img_folder, imgname)
            if os.path.isfile(imgpath):
                img_paths.append(imgpath)
            else:
                print('cannot find the image:', imgpath)

    return img_paths, mask_paths


if __name__ == '__main__':
    root = os.path.expanduser('~/.mxnet/datasets/mhp/LV-MHP-v2')
    split = 'train'
    tic = time.time()
    images_v1, masks_v1 = _get_mhp_pairs_v1(root, split)
    tic_v1 = time.time() - tic

    print('v1 results: ', len(images_v1), len(masks_v1))
    print('v1 running time: ', tic_v1)

    tic = time.time()
    images_v2, masks_v2 = _get_mhp_pairs_v2(root, split)
    tic_v2 = time.time() - tic

    print('v2 results: ', len(images_v2), len(masks_v2))
    print('v2 running time: ', tic_v2)