import os
from pathlib import Path

# Ultralytics YOLO 🚀, AGPL-3.0 license
from ultralytics import YOLO


"""
Detects objects in images using a YOLO model and saves the annotations in a specified directory.

Args:
    data (str): Path to the folder containing images for detection.
    output_dir (str | None, optional): Directory to save the annotation results.
        If None, a 'labels' folder will be created in the same directory as 'data'.
    det_model (str, optional): Pre-trained YOLO detection model. Defaults to 'best.pt'.
    device (str, optional): Device to run the model on. Defaults to 'CPU'.
"""


def detect_annotator(data='test/data/images', output_dir=None, det_model='best.pt', device='CPU'):
    # Initialize the YOLO detection model
    det_model = YOLO(det_model)

    # Set the path for data and create the output directory if not specified
    data_path = Path(data)
    if not output_dir:
        output_dir = data_path.parent / 'labels'
    os.makedirs(output_dir, exist_ok=True)

    # Perform object detection on the images
    det_results = det_model(data_path, stream=True, device=device)

    # Save annotations for each detected object
    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()
        boxes = result.boxes.xywhn

        img_filename = os.path.splitext(os.path.basename(result.path))[0]
        ann_filename = img_filename + '.txt'

        if len(class_ids):
            with open(os.path.join(output_dir, ann_filename), 'w') as f:
                for class_id, box in zip(class_ids, boxes):
                    box = map(str, box.tolist())
                    f.write(f'{class_id} ' + ' '.join(box) + '\n')


if __name__ == "__main__":
    ''' Uncomment and modify these lines if your values differ from the default ones '''
    # data_folder = ''
    # output_folder = ''
    # trained_model = ''
    # device_type = ''

    # Call the detect_annotator function with the specified parameters
    detect_annotator(
        ''' Uncomment and modify these lines if your values differ from the default ones '''
        # data_folder,
        # output_dir=output_folder,
        # det_model=trained_model,
        # device=device_type
    )
