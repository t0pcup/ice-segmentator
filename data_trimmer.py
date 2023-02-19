import numpy as np
import glob
from tqdm import trange

path = 'E:/files/data'
lst, ban = glob.glob(f'{path}/*.npy'), []

for i in trange(len(lst)):
    for j in range(i, len(lst)):
        if i != j and (np.load(lst[i]) == np.load(lst[j])).all():
            print(j, end=' ')
            ban.append((lst[i], lst[j]))

print(len(ban))
print(ban)
