# This module file contains information about each dataset used in this experiment with DETR.

COCO_CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "empty",
]

COCO_PEOPLE = [
    "N/A",
    "person",
]

PEOPLE_HQ = [
    "N/A",
    "person",
]

CHESS_PIECES_HQ = [
    "N/A",
    "chess_piece",
]


# --------------------------------------------------------

DATASET_CLASSES = {
    "coco": {
        "class_names": COCO_CLASSES,
        "empty_class_id": 91,
        "links": {
            "images": "http://images.cocodataset.org/zips/train2017.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        },
    },
    "coco_people": {
        "class_names": COCO_PEOPLE,
        "empty_class_id": 0,
        "links": {
            "roboflow": "https://universe.roboflow.com/shreks-swamp/coco-dataset-limited--person-only"
        },
    },
    "chess_pieces_hq": {
        "class_names": CHESS_PIECES_HQ,
        "empty_class_id": 0,
        "links": {
            "roboflow": "https://universe.roboflow.com/myroboflowprojects/chess-pieces-hq"
        },
    },
    "people_hq": {
        "class_names": PEOPLE_HQ,
        "empty_class_id": 0,
        "links": {
            "roboflow": "https://universe.roboflow.com/myroboflowprojects/people_hq"
        },
    },
}
