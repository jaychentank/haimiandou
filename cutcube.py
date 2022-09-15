import numpy as np
import csv
import SimpleITK as sitk
import os


def cutCube(X, center, shape, padd=0):  # center is a 3d coord (zyx)
    """
    给定图像数据，中心点位置，取块大小，返回指定大小的数据块，取块的策略是根据中心点，上下左右前后各取shape/2即可得到最终的数据块
    :param X: 三维的numpy数组，即输入的原始图像
    :param center: 一维的numpy数组，分别为[z,y,x,min_w,max_w]，zyx为对应的坐标，min_w为窗宽最小值，max_w为窗宽最大值
    :param shape: 一维的numpy数组类型，分别表示在 zyx(横断面、冠状面、矢状面）取块的层数
    :param padd: int类型，表示数据裁剪超出边界时的填充值
    :return: numpy的三维数组，返回根据中心位置提取的数据块
    """
    # 只取前三个数
    center = center[:3]
    # center = center.astype(int)
    # 每一维数据要取的层数为shape/2
    hlz = np.round(shape[0] / 2)
    hly = np.round(shape[1] / 2)
    hlx = np.round(shape[2] / 2)
    # 给超出边界的部分添加padding
    if ((center - np.array([hlz, hly, hlx])) < 0).any() or (
            (center + np.array([hlz, hly, hlx]) + 1) > np.array(
        X.shape)).any():
        Xn = np.ones(np.array(X.shape) + shape * 2) * padd
        Xn[shape[0]:(shape[0] + X.shape[0]), shape[1]:(shape[1] + X.shape[1]), shape[2]:(shape[2] + X.shape[2])] = X
        centern = center + shape
        cube = Xn[int(centern[0] - hlz):int(centern[0] - hlz + shape[0]),
               int(centern[1] - hly):int(centern[1] - hly + shape[1]),
               int(centern[2] - hlx):int(centern[2] - hlx + shape[2])]
        return np.copy(cube)
    else:
        cube = X[int(center[0] - hlz):int(center[0] - hlz + shape[0]),
               int(center[1] - hly):int(center[1] - hly + shape[1]),
               int(center[2] - hlx):int(center[2] - hlx + shape[2])]
        return np.copy(cube)


def load_mhd(path2scan):
    itkimage = sitk.ReadImage(path2scan)
    scan = sitk.GetArrayFromImage(itkimage)
    spacing = np.array(itkimage.GetSpacing())
    orientation = np.transpose(np.array(itkimage.GetDirection()).reshape((3, 3)))
    direction = itkimage.GetDirection()
    origin = np.flip(np.array(itkimage.GetOrigin()),
                     axis=0)  # origionally in yxz format (read xy in viewers but sanved as yx)
    return scan, spacing, orientation, origin, None, direction  # output in zyx format


def toNii(save_dir, file_name, img_array, pixel_spacing, origin, direction):
    savedImg = sitk.GetImageFromArray(img_array)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(pixel_spacing)
    # name = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '-' + file_name
    sitk.WriteImage(savedImg, os.path.join(save_dir, file_name))


if __name__ == '__main__':
    filename = '/mnt/data/xxyz/image-generation/mid_point_p.csv'
    cube_shape = [64, 64, 64]
    with open(filename) as f:
        f_csv = csv.reader(f)
        # print(f_csv)
        headers = next(f_csv)
        for row in f_csv:
            scan_path = '/mnt/data/xxyz/image-generation/new_pair_rotate/' + row[0] + '_Post_Rotate.mhd'
            print(scan_path)
            scan, spacing, orientation, origin, raw_slices, real_direction = load_mhd(scan_path)
            coord = [int(row[3]), int(row[2]), int(row[1])]
            x = cutCube(scan, coord, cube_shape, padd=0)
            toNii('/mnt/data/xxyz/image-generation/new_pair_resample_cutcube/', row[0] + '_Post.mhd', x, spacing,
                  origin, real_direction)

            scan_path = '/mnt/data/xxyz/image-generation/new_pair_rotate/' + row[0] + '_Pre_Rotate.mhd'
            print(scan_path)
            scan, spacing, orientation, origin, raw_slices, real_direction = load_mhd(scan_path)
            coord = [int(row[3]), int(row[2]), int(row[1])]
            x = cutCube(scan, coord, cube_shape, padd=0)
            toNii('/mnt/data/xxyz/image-generation/new_pair_resample_cutcube/', row[0] + '_Pre.mhd', x, spacing,
                  origin, real_direction)