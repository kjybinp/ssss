import os

def prepare(dire,output):
    f = open(output,'w')
    s = ''
    f.write(s)
    f.close()
    cwd = os.getcwd()
    for i in os.listdir(dire):
        f = open(output,'a')
        s = cwd + '/' + dire + '/' + i + '\n'
        f.write(s)

prepare('images/train_images','train.txt')
prepare('images/test_images','test.txt')