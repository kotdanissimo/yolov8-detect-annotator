import os
from pathlib import Path

from ultralytics import YOLO


# Function to annotate detections from YOLO model
def detect_annotator(
        data: str = 'data/images',
        output_dir: str = 'output',
        det_model: str = 'data/best.pt',
        used_device: dict = 'CPU'
):

    """
    This function takes in the path to the data, the output directory, the detection model, and the device to run the model on.
    It then runs the model on the data and writes the detection results to the output directory.

    :param data: str, path to the data to be annotated, default is 'test/data/images'
    :param output_dir: str, path to the directory where the annotated data will be saved, default is None
    :param det_model: str, path to the trained YOLO model, default is 'best.pt'
    :param used_device: str, device to run the model on, default is 'CPU' or int, to use GPU
    :return: None
    """

    # Load the YOLO model
    det_model = YOLO(det_model)

    # Set the data path and output directory
    data_path = Path(data)
    if not output_dir:
        output_dir = data_path.parent / 'output'

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Run the model on the data
    det_results = det_model(
        data_path,
        stream=True,
        device=used_device
    )

    # Iterate over the detection results
    for result in det_results:
        # Get the class IDs and bounding boxes
        class_ids = result.boxes.cls.int().tolist()
        boxes = result.boxes.xywhn

        # Get the image filename and create the annotation filename
        img_filename = os.path.splitext(os.path.basename(result.path))[0]
        ann_filename = img_filename + '.txt'

        # If there are detections, write them to the annotation file
        if len(class_ids):
            with open(os.path.join(output_dir, ann_filename), 'w') as f:
                for class_id, box in zip(class_ids, boxes):
                    box = map(str, box.tolist())
                    f.write(f'{class_id} ' + ' '.join(box) + '\n')


if __name__ == "__main__":
    detect_annotator()
