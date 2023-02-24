import h5py
import json
import random


def process_item(i):
    item = json.loads(i)
    d = item["article"]
    if len(d) > 5:
        return json.dumps(d)
    else:
        return None


cd = h5py.File("data/CNN_DM/cd.train.h5df", "r")['dataset']
nyt = h5py.File("data/NYT/nyt.train.h5df", "r")['dataset']

new_dataset = []
print("start processing cd dataset")
for data in cd:
    doc = process_item(data)
    if doc is not None:
        new_dataset.append(doc)
print("finish processing cd dataset")

print("start processing nyt dataset")
for data in nyt:
    doc = process_item(data)
    if doc is not None:
        new_dataset.append(doc)
print("finish processing nyt dataset")

print("start saving data")

news = h5py.File("data/train.h5df", "w")
news_dataset = news.create_dataset("dataset", (len(new_dataset), ), dtype=h5py.special_dtype(vlen=str))

random.shuffle(new_dataset)
news_dataset[:] = new_dataset
print("finish saving data")
print("training data has", len(new_dataset), "documents")
