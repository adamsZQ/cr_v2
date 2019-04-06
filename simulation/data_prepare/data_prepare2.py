import json
''' 
    delete movie_data_cleaned_normal 
'''
data_named_list = []
id_list = []
with open('/home/next/cr_repo/movie_data_cleaned_normal') as f:
    for line in f:
        data_named = json.loads(line)
        data_named_list.append(data_named)
        id_list.append(int(data_named['id']))


data_id_list = []
id_list_2 = []
with open('/home/next/cr_repo/entity.dat') as w:
    for line in w:
        data_id = json.loads(line)
        data_id_list.append(data_id)
        id_list_2.append(int(data_id['id']))


data_named_final_list = []
for data_id in data_id_list:
    for data_named in data_named_list:
        data_named['id'] = int(data_named['id'])
        if data_named['id'] == data_id['id']:
            data_named_final_list.append(data_named)
            break

print(len(data_named_final_list))
print(len(data_id_list))

for movie_named, movie_id in zip(data_named_final_list, data_id_list):
    if movie_named['id'] != movie_id['id']:
        print(movie_named['id'])
        print(movie_id['id'])


with open('/home/next/cr_repo/movie_data_cleaned2', 'w') as t:
    for line in data_named_final_list:
        line = json.dumps(line)
        t.write(line)
        t.write('\n')


# for data_named in data_named_list:
#     if int(data_named['id']) not in id_list_2:
#         id_list.pop(data_named_list.index(data_named))
#         data_named_list.remove(data_named)
#
# data_named_list.pop(id_list.index(2896))
# data_named_list.pop(id_list.index(64918))
#
# id_list.remove(2896)
# id_list.remove(64918)
#
# print(len(data_named_list))
# print(len(data_id_list))
# print(len(id_list))
# print(set(id_list) - set(id_list_2))
#

# # id_list_3 = []
# # with open('/home/next/cr_repo/movie_rating') as w:
# #     for line in w:
# #         data_rat = json.loads(line)
# #         # data_id_list.append(data_id)
# #         if int(data_rat['movie']) not in id_list_3:
# #             id_list_3.append(int(data_rat['movie']))
# #
# # print(len(id_list_3))
# # print(set(id_list_3) - set(id_list_2))
# #
