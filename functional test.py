import SimpleITK as sitk
import numpy as np

pre_mhd = sitk.ReadImage("/mnt/data/xxyz/image-generation/new pair data/liangxingwang_Pre.mhd")
pre_array = sitk.GetArrayFromImage(pre_mhd)
# pre_array = [[[0]]]
# pre_array = np.array(pre_array)

post_mhd = sitk.ReadImage("/mnt/data/xxyz/image-generation/new pair data/liangxingwang_Post.mhd")
post_array = sitk.GetArrayFromImage(post_mhd)
# post_array = [[[1]]]
# post_array = np.array(post_array)

print(pre_array.shape)
print(pre_array)
print(post_array)
# diff = []
# for i in range(pre_array.shape[0]):
#     for j in range(pre_array.shape[1]):
#         for k in range(pre_array.shape[2]):
#             if pre_array[i][j][k] != post_array[i][j][k]:
#                 diff.append(list([i, j, k]))
#
# print(diff)

tmp = pre_array != post_array
print(tmp)
