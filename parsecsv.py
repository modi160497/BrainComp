import csv
import re
import numpy
from itertools import islice

def parse():

    csvpath = "trainingtwitter.csv"


    fd = open(csvpath, "r")

    dataset_reader = csv.reader(fd,delimiter="|",quotechar='"')

    i = 0

    sentiment1 = []
    sentiment2 = []

    for row in dataset_reader:
        if i == 0:
            i = i + 1
            continue
        s = row[0]
        splitsen = s.split(",", 1)
        if(splitsen[0]=='0'):
            sentiment1.append((0,splitsen[1].lower()))
        if (splitsen[0]=='4'):
            sentiment2.append((1,splitsen[1].lower()))

    print(len(sentiment1))
    print(len(sentiment2))
    # list of length in which we have to split
    length_to_split = [400000,100000]
    length_to_split2 = [100000, 20000]
    length_to_splittrain1 = [50000]

    # Using islice
    Input = iter(sentiment1)
    Output = [list(islice(Input, elem))
              for elem in length_to_split]

    Input2 = iter(sentiment2)
    Output2 = [list(islice(Input2, elem))
              for elem in length_to_split2]

    Input3 = iter(sentiment1)
    Output3 = [list(islice(Input2, elem))
               for elem in length_to_splittrain1]

    Input4 = iter(sentiment2)
    Output4 = [list(islice(Input2, elem))
               for elem in length_to_split]

    print(sentiment1[0])
    print(sentiment2[0])

    trainword = Output[0] + Output2[0]

    test = Output[1] + Output2[1]

    train = Output3[0] + Output4[0]

    return trainword, test, train

