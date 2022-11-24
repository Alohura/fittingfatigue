import collections
import pickle

import numpy as np
import pandas as pd

from util_functions import *
from scipy.integrate import quadrature
import numpy as np

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

# path = r"C:\code\test"
# file_name="rashila.txt"
# f = open(f"{path}\\{file_name}",'r')
# dummy_variable = f.readlines()
# f.close()
# print(dummy_variable)
# f = open(f"{path}\\{file_name}",'w')
# f.write("hello!!")
# f.close()

d_id = 24. / 1000.
d_od = 48. / 1000.
r_notch = 3.5 / 1000.
f_axial = 100000.
m_bending = 300.

# stress = stress_stem_roark_17(f_axial, m_bending, d_od, d_id, r_notch)
scf_a = scf_roark_17a(d_od, d_id, r_notch)
scf_b = scf_roark_17b(d_od, d_id, r_notch)
print("SCFs [Axial, bending]:", [scf_a, scf_b])
a = 1

n_deltas, n_stresses = stress_histogram_en_1991_1_4(100, 2, 3)
b = sum(n_deltas) / 1e9
a = 1

list1 = [1, 2, 24]
list2 = [1, 2, 23]

return_list = [[x, list2[list1.index(x)]] for x in list1 if x not in list2]
print(return_list)



my_dict = {'180': {'m1': 3, 'm2': 5, 's1': 180, 's2': 133, 's3': 73, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '95-100', 'ca_list': [95.0, 100.0]}, '160': {'m1': 3, 'm2': 5, 's1': 160, 's2': 118, 's3': 65, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '75-95', 'ca_list': [75.0, 95.0]}, '140': {'m1': 3, 'm2': 5, 's1': 140, 's2': 103, 's3': 57, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '55-75', 'ca_list': [55.0, 75.0]}, '125': {'m1': 3, 'm2': 5, 's1': 125, 's2': 92, 's3': 51, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '35-55', 'ca_list': [35.0, 55.0]}, '112': {'m1': 3, 'm2': 5, 's1': 112, 's2': 83, 's3': 45, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '15-35', 'ca_list': [15.0, 35.0]}, '100': {'m1': 3, 'm2': 5, 's1': 100, 's2': 74, 's3': 40, 'n_s1': 2000000, 'n_cut_off': 100000000, 'ca': '0-15', 'ca_list': [0.0, 15.0]}}
print(sn_from_ca_values(16., my_dict))

print(max("2a", "1d"))
print(max(2, np.nan))

my_list = [1, 2, [2, 3]]
new_list = [x for x in my_list if type(x) is not list] + [x[:] for x in my_list if type(x) is list][0]
print(new_list)
print("12".split(","))

lst1 = [1,2,3]
lst2 = [3,5,6]
lst3 = [7,8,9]

print(sum([0., 0.]))

my_list = np.array([1,2,3,4,5,6,5,7,8,6,9,9,10,1])
print(np.unique(my_list), np.nonzero(my_list==5), (my_list==5).nonzero())
print(my_list[np.nonzero(my_list == 5)])
b=(my_list==5).nonzero()
my_frame = pd.DataFrame(data=[[1,2,3], [4,5,6], [7,8,9]], columns=["a", "b", "c"])

print(my_frame.iloc[[0, 2], [0, 1]])
rows = np.array(my_frame.index)
index_lists = [list((rows == x).nonzero()[0]) for x in np.unique(rows)]

lst2 = np.array(['851B', '7D'])
lst1 = ['11D', '851B']
# a1 = check_overlap_between_two_lists(lst1, lst2)

set1, set2 = set(lst1), set(lst2)
a2 = set1 - set2
a3 = set2 - set1

a = 0 if len(a2) == len(set1) else 1

# 'npl-sfd-a'['11D', ' 851B']

b=' 851'
print(len(b.strip()))

print(str(4.0))

my_dict = {
       "item1": 1,
       "item2": "text"
}
# pickle.dump(my_dict, open(r"C:\Users\AndreasLem\Downloads\pdump.p", "wb"))
# # pickle.dump(my_dict, "pdump")
#
# my_dict2 = pickle.load(open(r"C:\Users\AndreasLem\Downloads\pdump.p", "rb"))
# a = 1

print(max([-2,1], key=abs))


def integral_function(theta, my):
       return my / (my * np.tan(theta) + 1.)


mys = [i / 10. for i in range(11)]
print(mys)
print("Integral: ", [quadrature(integral_function, np.arcsin(13. / 27.), np.arcsin(19. / 27.), args=(my))[0] for my in mys])

lst = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
org_lst = [
       ['j', 'b', False],
       ['l', 'a', True],
       ['a', 'l', False],
       ['b','j',False]
]

new_lst = list_items_move(lst, org_lst)
print(lst)
print(new_lst)

columns = [
       'row', 'structure_number', 'lc', 'wc',
       'lc_description', 'set_no', 'phase_no', 'joint', 'vertical',
       'transversal', 'longitudinal', 'line_id', 'resultant', 't1', 't2',
       't_friction', 'm_section', 'SCF_axial', 'SCF_bending', 'stress_axial',
       'stress_bending', 'stress', 'stress_axial_range',
       'stress_bending_range', 'stress_range', 'f_long_nom', 'f_trans_nom',
       'f_vert_nom', 'swing_angle_trans', 'swing_angle_trans_orig',
       'swing_angle_trans_range', 'swing_angle_long', 'swing_angle_long_orig',
       'swing_angle_long_range', 'sn_curve', 'damage'
]


columns = list_items_move(
       columns,
       [
              ["line_id", "structure_number", False],
              ["set_no", "lc", False],
              ["resultant", "phase_no", True],
              ["joint", "resultant", False],
       ]
)
# print(columns)

print(scf_roark_17b(40.2,20.35,1))