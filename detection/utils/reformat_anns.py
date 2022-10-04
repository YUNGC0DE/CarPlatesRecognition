import json
import os
from tqdm import tqdm
import xmltodict
from glob import glob


base_path = "/home/evgenii/Desktop/ml_hw/CarPlates"
files = glob(os.path.join(base_path, "annotations/*"))
os.makedirs(os.path.join(base_path, "ann_json"), exist_ok=True)

for file in tqdm(files):
    file_name = file.split('/')[-1]
    save_path = os.path.join(base_path, "ann_json", file_name.replace(".xml", ".json"))
    with open(file) as xml_file:
        new_ann = []
        json_data = xmltodict.parse(xml_file.read())
        annotations = json_data["annotation"]["object"]
        if not isinstance(annotations, list):
            annotations = [annotations]
        for an in annotations:
            new_ann.append({
                "class": 2,
                "box": list(map(int, an["bndbox"].values()))
            })
        print(new_ann)
        with open(save_path, "w") as json_file:
            json_file.write(json.dumps(new_ann))
