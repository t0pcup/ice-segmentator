a = ['00_-9', '01_99', '02_10', '10_99', '20_-9', '20_03', '20_04',
     '20_05', '20_99', '30_03', '30_04', '30_05', '30_99', '50_03',
     '50_04', '50_05', '50_06', '50_99', '60_03', '70_03', '70_04',
     '70_06', '70_07', '80_03', '80_04', '80_05', '90_03', '90_04',
     '90_05', '90_06', '90_99', '91_03', '91_04', '91_05', '91_06',
     '91_08', '92_03', '92_04', '92_08']
b = [1042, 995, 565, 4, 32, 133, 12, 15, 103, 40, 8,
     81, 44, 10, 21, 15, 18, 35, 6, 110, 15, 15,
     14, 77, 25, 27, 68, 35, 148, 44, 3, 46, 98,
     119, 436, 6, 17, 20, 1330]

# '80',
# high = ['80', '90', '91', '92']
high = []
form_l = ['01', '02', '03', '04']
form_h = ['06', '07', '05']
print([(a[i], b[i]) for i in range(len(a)) if a[i][-2:] in form_l and a[i][:2] in high])
print(sum([b[i] for i in range(len(a)) if a[i][-2:] in form_l and a[i][:2] in high]))
print([(a[i], b[i]) for i in range(len(a)) if a[i][-2:] in form_l and a[i][:2] not in high])
print(sum([b[i] for i in range(len(a)) if a[i][-2:] in form_l and a[i][:2] not in high]))

print()

# print([(a[i], b[i]) for i in range(len(a)) if a[i][-2:] in form_h and a[i][:2] in high])
# print(sum([b[i] for i in range(len(a)) if a[i][-2:] in form_h and a[i][:2] in high]))
# print([(a[i], b[i]) for i in range(len(a)) if a[i][-2:] in form_h and a[i][:2] not in high])
# print(sum([b[i] for i in range(len(a)) if a[i][-2:] in form_h and a[i][:2] not in high]))
print([(a[i], b[i]) for i in range(len(a)) if a[i][-2:] in form_h and a[i][:2] not in []])
print(sum([b[i] for i in range(len(a)) if a[i][-2:] in form_h and a[i][:2] not in []]))


# import warnings
# import glob
# import geopandas as gpd
# from tqdm import tqdm
# import numpy as np
# import fiona
#
# translate_classes_simple = {
#     '55': 0,  # ice free
#     '01': 0,  # open water
#
#     '02': 1,  # bergy water | FA 10 | icebergs
#
#     '10': 0,  # 1/10 | noname FA
#     '12': 0,  # 1-2/10
#     '13': 0,  # 1-3/10
#
#     '20': 2,  # 2/10 | FA 03-05
#     '23': 2,  # 2-3/10
#     '24': 2,  # 2-4/10
#     '30': 2,  # 3/10 | FA 03-05
#     '34': 2,  # 3-4/10
#     '35': 2,  # 3-5/10
#     '40': 2,  # 4/10 | ???
#     '45': 2,  # 4-5/10
#     '46': 2,  # 4-6/10
#     # ++++++++++++++
#     '50': 2,  # 5/10 | FA 03-05
#     '56': 2,  # 5-6/10
#     '57': 2,  # 5-7/10
#     '60': 2,  # 6/10 | ???
#     '67': 2,  # 6-7/10
#     '68': 2,  # 6-8/10
#
#     '70': 3,  # 7/10 | FA 03-07
#     '78': 3,  # 7-8/10
#     '79': 3,  # 7-9/10
#     '80': 3,  # 8/10 | FA 03-05
#     '89': 3,  # 8-9/10
#     '81': 3,  # 8-10/10
#     '90': 3,  # 9/10 | FA 03-06
#
#     '91': 4,  # 9-10/10 | FA 03-08 TODO
#     '92': 4,  # 10/10 | FA 03-08 TODO
# }
# codes = []
# dic = dict(zip(translate_classes_simple.keys(),
#                [[] for _ in translate_classes_simple.keys()]))
#
#
# def def_num(it: dict) -> int:
#     # undefined / land / no data => zero
#     trans_dict = {'L': 0, 'W': 0, 'N': 0, 'S': 5}  # no shelves in src
#     try:
#         ct, fa = int(it['CT']), int(it['FA'])
#         FA_stat.append(it['FA'])
#         CT_stat.append(it['CT'])
#         CT_FA_stat.append(it['CT'] + '_' + it['FA'])
#     except:
#         _ = 0
#     try:
#         # print(it['POLY_TYPE'], it['CT'], it['FA'])
#         if it['FA'] not in dic[it['CT']]:
#             dic[it['CT']].append(it['FA'])
#     except:
#         _ = 0
#         # print(it['POLY_TYPE'], it['CT'], it['FA'])
#
#     try:
#         return trans_dict[it['POLY_TYPE']]
#     except KeyError:
#         codes.append(f"\tCT: {it['CT']} FA: {it['FA']}")
#         # print(f"\tCT: {it['CT']} FA: {it['FA']}")
#         # if it['FA'] in ['08', '09', '10']:
#         #     codes.append(f"\tCT: {it['CT']} FA: {it['FA']}")
#
#         if it['FA'] == '08':
#             return 5
#         try:
#             return translate_classes_simple[it['CT']]
#         except KeyError:
#             print(it['POLY_TYPE'], it['CT'], it['FA'])
#             return -1
#
#
# warnings.filterwarnings("ignore")
# reg_path = 'C:/files/regions/2021'
# class_stat = []
# FA_stat, CT_stat, CT_FA_stat = [], [], []
# l_ = list(np.random.permutation(glob.glob(f'{reg_path}/*2021*.shp')))[:20]
# for f in tqdm(l_):
#     shapes_list = gpd.read_file(f).to_crs('epsg:4326')
#     file = fiona.open(f)
#
#     geom_value = []
#     for i in range(len(file)):
#         prop = file[i]
#         class_stat.append(def_num(prop['properties']))
#
# # print(dic)
# # print(np.unique(np.asarray(class_stat), return_counts=True))
# print(np.unique(np.asarray(CT_stat), return_counts=True))
# print(np.unique(np.asarray(FA_stat), return_counts=True))
# print(np.unique(np.asarray(CT_FA_stat), return_counts=True))
