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

# sn_curve = {
#     180: {'m1': 3, 'm2': 5, 's1': 180, 's2': 133, 's3': 73, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '100-95'},
#     160: {'m1': 3, 'm2': 5, 's1': 160, 's2': 118, 's3': 65, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '95-75'},
#     140: {'m1': 3, 'm2': 5, 's1': 140, 's2': 103, 's3': 57, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '75-55'},
#     125: {'m1': 3, 'm2': 5, 's1': 125, 's2': 92, 's3': 51, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '55-35'},
#     112: {'m1': 3, 'm2': 5, 's1': 112, 's2': 83, 's3': 45, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '35-15'},
#     100: {'m1': 3, 'm2': 5, 's1': 100, 's2': 74, 's3': 40, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '15-0'}
# }
# sn_curve = {f"{x}": y for x, y in sn_curve.items()}

