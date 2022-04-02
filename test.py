import numpy as np
from util_functions import *

a = [1, 2]
b = [3, 4]
c = np.array(a) + np.array(b)
print(c)
c = a + b
print(c)
mydict = {"a": 1, "b": 2}
tmp = "10c"
tmp_lst = split_word_to_list(tmp)
a = list(set(tmp_lst).intersection(mydict.keys()))

d = 1
b = 1

path = r"C:\code\test"
file_name="rashila.txt"
f = open(f"{path}\\{file_name}",'r')
dummy_variable = f.readlines()
f.close()
print(dummy_variable)
# f = open(f"{path}\\{file_name}",'w')
# f.write("hello!!")
# f.close()

d_id = 24. / 1000.
d_od = 50. / 1000.
r_notch = 1. / 1000.
f_axial = 100000.
m_bending = 300.

stress = stress_stem_roark_17(f_axial, m_bending, d_od, d_id, r_notch)
a = 1

n_deltas, n_stresses = stress_histogram_en_1991_1_4(100, 2, 3)
b = sum(n_deltas) / 1e9
a = 1
