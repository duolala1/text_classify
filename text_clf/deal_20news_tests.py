import os
import re

'''生成原始语料文件夹下文件列表'''
def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
                listdir(file_path, list_name)
        else:
                list_name.append(file_path)

list_name = []
listdir('data/20-news/',list_name)
# print(list_name)
for path in list_name[1:2]:
    print(path)
    f = open(path)
    text = ""
    text = f.read()
    # text = text + line
    # while line:
    #     line = f.readline()
    #     line = line
    #     text = text + line
    text.replace('\n', '')
    print(text + '\n')

    f.close()