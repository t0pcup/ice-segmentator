import numpy as np
import os
from numpy import copy


def unique_classes(array):
    names = []
    for ind in np.unique(array):
        if ind not in names:
            names.append(ind)

    count = np.unique(array, return_counts=True)

    dct = dict.fromkeys(names)
    for val in dct:
        dct[val] = 0
    for ind in range(len(count[0])):
        dct[count[0][ind]] += count[1][ind]
    return dct


path = r'E:/dafuck/label'

for i in os.listdir(path):
    arr = np.load(f'{path}/{i}')
    dd = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}
    new_arr = copy(arr)
    for k, v in dd.items():
        new_arr[arr == k] = v

    # print(unique_classes(arr))
    # print(unique_classes(new_arr))

    with open(f'{path}0/{i}', 'wb') as f:
        np.save(f, new_arr)


# path = r'C:/AURORA/NorthenSeaRoute/ICE/stage_2/label0'
#
# nms = []
# for i in os.listdir(path):
#     for j in np.unique(np.load(f'{path}/{i}')):
#         if j not in nms:
#             nms.append(j)
#
# cnt = [np.unique(np.load(f'{path}/{i}'), return_counts=True) for i in os.listdir(path)]
# d = dict.fromkeys(nms)
# for v in d:
#     d[v] = 0
#
# for i in cnt:
#     for j in range(len(i[0])):
#         d[i[0][j]] += i[1][j]
#
# print(d)

#
# import numpy as np
# import os
# from numpy import copy
#
# path = r'C:/AURORA/NorthenSeaRoute/ICE/stage_2/data'
# path_label = r'C:/AURORA/NorthenSeaRoute/ICE/stage_2/label01'
# print(np.load(f'{path}/{os.listdir(path)[0]}').shape)
# print(np.load(f'{path}/{os.listdir(path_label)[0]}').shape)
# mins, maxs, means, std = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
# stf = np.array([])
#
# for i in os.listdir(path):
#     arr = np.load(f'{path}/{i}')
#     if i == os.listdir(path)[0]:
#         stf = arr
#     else:
#         stf = np.concatenate((stf, arr))
#
#     for j in range(4):
#         mins[j].append(np.min(arr[j]))
#         maxs[j].append(np.max(arr[j]))
#         means[j].append(np.mean(arr[j]))
#
# print(sorted(mins))
# print(sorted(means))
# print(sorted(maxs))
# print(np.std(stf))
#
# # min = -84.64701
# # max = 4.379311
#
#
# import os
# import numpy as np
# path_label = r'C:/AURORA/NorthenSeaRoute/ICE/stage_2/label01'
# path = r'C:/AURORA/NorthenSeaRoute/ICE/stage_2/data'
# for i in os.listdir(path_label):
#     if i not in os.listdir(path+'1'):
#         np.load(f'{path}/{i}')
#         np.save(path+f'1/{i}', np.load(f'{path}/{i}'))
#
# print('+')
