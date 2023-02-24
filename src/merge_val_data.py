import h5py
import json
import random


def process_item(i):
    item = json.loads(i)
    d = item["article"]
    a = item["abstract"]
    if len(d) > 0 and len(a) > 0:
        return json.dumps({"article": d, "abstract": a})
    else:
        return None


cd = h5py.File("data/CNN_DM/cd.validation.h5df", "r")['dataset']
nyt = h5py.File("data/NYT/nyt.validation.h5df", "r")['dataset']

cd_dataset = []
nyt_dataset = []
print("start processing cd dataset")
for data in cd:
    doc = process_item(data)
    if doc is not None:
        cd_dataset.append(doc)
print("finish processing cd dataset")

print("start processing nyt dataset")
for data in nyt:
    doc = process_item(data)
    if doc is not None:
        nyt_dataset.append(doc)
print("finish processing nyt dataset")

new_dataset = random.sample(cd_dataset, 500) + random.sample(nyt_dataset, 500)

print("start saving data")

news = h5py.File("data/val.h5df", "w")
news_dataset = news.create_dataset("dataset", (len(new_dataset), ), dtype=h5py.special_dtype(vlen=str))

news_dataset[:] = new_dataset
print("finish saving data")
print("val data has", len(new_dataset), "documents")
