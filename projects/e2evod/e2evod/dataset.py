import xml.etree.ElementTree as ET

from detectron2.data import DatasetCatalog, MetadataCatalog


def get_vid_dicts(data_dir):
    meta = MetadataCatalog.get("vid")
    dataset_dicts = []
    vid_dir = os.path.join(data_dir, "VID")
    for video in os.listdir(vid_dir):
        records = []
        jpegs = os.listdir(os.path.join(vid_dir, video, "Data"))
        xmls = os.listdir(os.path.join(vid_dir, video, "Annotations"))
        for jpeg, xml in zip(jpegs, xmls):
            record = {}
            record["filename"] = os.path.join(vid_dir, video, "Data", jpeg)
            tree = ET.parse(os.path.join(vid_dir, video, "Annotations", xml))
            record["height"] = int(tree.find("size").find("height").text)
            record["width"] = int(tree.find("size").find("width").text)

            boxes, classes, trackids = [], [], []
            objects = tree.findall("object")
            for obj in objects:
                if not obj.find("name").text in meta.classes_to_ind:
                    continue

                bbox = obj.find("bndbox")
                box = [
                    float(bbox.find("xmin").text), float(bbox.find("ymin").text),
                    float(bbox.find("xmax").text), float(bbox.find("ymax").text)
                ]
                boxes.append(box)
                classes.append(
                    meta.classes_to_ind[obj.find("name").text.lower().strip()]
                )
                trackids.append(int(obj.find("trackid").text))
            
            record["boxes"] = boxes
            record["classes"] = classes
            record["trackids"] = trackids
            records.append(record)
        dataset_dicts.append(records)

    # use IMAGENET DET data
    if "train" in data_dir:
        det_dir = os.path.join(data_dir, "DET")
        jpegs = os.listdir(os.path.join(det_dir, "Data"))
        xmls = os.listdir(os.path.join(det_dir, "Annotations"))
        for jpeg, xml in zip(jpegs, xmls):
            record = {}
            record["filename"] = os.path.join(det_dir, "Data", jpeg)
            tree = ET.parse(os.path.join(det_dir, "Annotations", xml))
            record["height"] = int(tree.find("size").find("height").text)
            record["width"] = int(tree.find("size").find("width").text)

            boxes, classes, trackids = [], [], []
            objects = tree.findall("object")
            for i, obj in enumerate(objects):
                if not obj.find("name").text in meta.classes_to_ind:
                    continue

                bbox = obj.find("bndbox")
                box = [
                    float(bbox.find("xmin").text), float(bbox.find("ymin").text),
                    float(bbox.find("xmax").text), float(bbox.find("ymax").text)
                ]
                boxes.append(box)
                classes.append(
                    meta.classes_to_ind[obj.find("name").text.lower().strip()]
                )
                trackids.append(i)

            record["boxes"] = boxes
            record["classes"] = classes
            record["trackids"] = trackids
            dataset_dicts.append([record])

    return dataset_dicts


def register_vid_instances(name, metadata, image_root):
    """
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: globals()[f"get_vid_dicts"](image_root))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        image_root=image_root, evaluator_type="vid", **metadata
    )
