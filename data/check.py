fi = open("train-images-idx3-ubyte.nak", 'rb')
magic = fi.read(4)
print magic.encode('hex')
num = fi.read(4)
print int(num.encode('hex'), 16)
num = fi.read(4)
print int(num.encode('hex'), 16)
num = fi.read(4)
print int(num.encode('hex'), 16)

fl = open('train-labels-idx1-ubyte.nak', 'rb')
magic = fl.read(4)
print magic.encode('hex')
num = fl.read(4)
print int(num.encode('hex'), 16)

print

dic = {}
for i in range(0, 60000):
    dic[fi.read(784)] = int(fl.read(1).encode('hex'), 16)

print

import csv

reader = csv.reader(open('../sample.csv'))
writer = csv.writer(open('result.csv', 'w'))

lines = [l for l in reader]
new_lines = []
for filename, n0,n1,n2,n3,n4,n5,n6,n7,n8,n9 in lines:
    new_lines.append([filename])
    
file_index = 0
for filename, n0,n1,n2,n3,n4,n5,n6,n7,n8,n9 in lines:
    ftest = open('../test/' + filename, 'rb')
    rtest = ftest.read()

    new_lines[file_index].append(dic[rtest])
    file_index += 1
    
    if file_index % 1000 == 0:
        print file_index
        
writer.writerows(new_lines)