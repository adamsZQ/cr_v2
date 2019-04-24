

# a_list = [1,2,3]
# b_list = ['a', 'b', 'c']
# c_list = ['!', '@', '#']
#
# zipped = zip(a_list, b_list, c_list)
#
# print(zipped)

# a = 'sadf'
# b = []
# b.append(a)
# a = '123'
# b.append(a)
# print(b)

# a = [1,2,[3,4]]
#
# print(len(a))
# for i in a:
#     if i !=1 :
#         print(i)

# a = [1,2,3]
# b = ['a']
# b = a[:3]
#
# print(b)

# a = 'sdf_d'
# v = a.split('_')[1]
# print(v)
import numpy as np

aa = [2, 3]

ttt = []
ttt.append([js for js in aa])
dididi = np.reshape(aa, (1, 3, 1)).tolist()
print(dididi)