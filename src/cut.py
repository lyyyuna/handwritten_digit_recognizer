import csv

reader = csv.reader(open('../output.csv'))
writer = csv.writer(open('../submit0413.csv', 'w'))
lines = [l for l in reader]
new_lines = []
for filename, n0,n1,n2,n3,n4,n5,n6,n7,n8,n9 in lines:
    # path = '../test/' + filename
    # x = numpy.fromfile(path, dt)
    # x = numpy.reshape(x, (784, 1))
    # result = net.feedforward(x)

    line = [filename,
            float('%0.4f' % float(n0)),
            float('%0.4f' % float(n1)),
            float('%0.4f' % float(n2)),
            float('%0.4f' % float(n3)),
            float('%0.4f' % float(n4)),
            float('%0.4f' % float(n5)),
            float('%0.4f' % float(n6)),
            float('%0.4f' % float(n7)),
            float('%0.4f' % float(n8)),
            float('%0.4f' % float(n9))
            ]
    new_lines.append(line)


writer.writerows(new_lines)
