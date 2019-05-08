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

    #print(len(sentiment1))
    #print(len(sentiment2))
    # list of length in which we have to split
    length_to_split = [400000,100000]
    length_to_split2 = [100000, 20000]
    length_to_splittrain1 = [5]

    # Using islice
    Input = iter(sentiment1)
    Input2 = iter(sentiment2)


    Output = sentiment1[:400000]
    Output2 = sentiment2[:100000]


    Output3 = sentiment1[:5]

    Output4 = sentiment2[:5]

    #print(sentiment1[0])
    #print(sentiment2[0])

    trainword = Output + Output2

    test = Output + Output2

    train = Output3 + Output4

    print("hello")
    print(train)

    return trainword, test, train

