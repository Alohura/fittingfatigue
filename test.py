import numpy as np
from util_functions import split_word_to_list

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
