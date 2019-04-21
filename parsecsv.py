import csv

def parse():
    csvpath = "py_isear_dataset\isear.csv"

    fd = open(csvpath, "r")

    dataset_reader = csv.reader(fd,delimiter="|",quotechar='"')

    i = 0

    emotions = []
    sentences = []

    for row in dataset_reader:
        if i == 0:
            i = i + 1
            continue
        emotions.append(row[36])
        sentences.append(row[40])

    #print(emotions)
    #print(sentences)

    return emotions,sentences