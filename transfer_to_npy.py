import os.path
import random

import numpy
import numpy as np
import SimpleITK as sitk
import csv


def load_mhd(path2scan):
    itkimage = sitk.ReadImage(path2scan)
    scan = sitk.GetArrayFromImage(itkimage)
    spacing = np.array(itkimage.GetSpacing())
    orientation = np.transpose(np.array(itkimage.GetDirection()).reshape((3, 3)))
    direction = itkimage.GetDirection()
    origin = np.flip(np.array(itkimage.GetOrigin()),
                     axis=0)  # origionally in yxz format (read xy in viewers but sanved as yx)
    return scan, spacing, orientation, origin, None, direction  # output in zyx format


if __name__ == '__main__':
    data_dir = '/mnt/data/xxyz/image-generation/new_pair_resample_cutcube/'
    # data_dir = '/mnt/data/czy/haimiandou/data/sinus_modified/'
    out_dir_A = './data/trainA'
    out_dir_B = './data/trainB'
    shape = [64, 64, 64]
    # f = csv.reader(open('/mnt/data/czy/haimiandou/mid_point.csv', 'r'))
    f = csv.reader(open('/mnt/data/xxyz/image-generation/mid_point_p.csv', 'r'))
    for i in f:
        if i[0] == 'name':
            continue
        # 无显影的路径
        # scan_dir = os.path.join(data_dir, '%s_cube.mhd' % (i[0]))
        scan_dir = os.path.join(data_dir, '%s_Pre_Rotate.mhd' % (i[0]))

        if os.path.exists(scan_dir):
            print('[*]path : ', scan_dir)
            center_zyx = np.array([i[3], i[2], i[1]])

            scan, spacing, orientation, origin, raw_slices, real_direction = load_mhd(scan_dir)

            # scan_cube = cutCube(scan,center_zyx,shape,padd = 0)

            filename_A = os.path.join(out_dir_A, '%s_Pre.npy' % i[0])
            np.save(filename_A, scan)
            # filename_B = os.path.join(out_dir_B, '%s_Post.npy' % i[0])
            # np.save(filename_B, scan)

            # if (change_cube == scan_cube).all():
            #     print('[!] not change!')
