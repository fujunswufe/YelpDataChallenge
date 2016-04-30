__author__ = 'fujun'

path = '/Users/fujun/Downloads/output/system/'

import os

filenames = os.listdir(path)

for name in filenames:
    print(name)
    prefix = name[0: name.rindex('.')]
    #print(prefix)
    new_name = 'task'+prefix+'_englishSyssum'+prefix+'.txt'
    os.rename(path+name, path+new_name)