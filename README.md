## About 

YOLOv8 Detect Annotator is a tool designed for automated image annotation using a pre-trained YOLOv8 model, streamlining object detection and annotation tasks to facilitate further training.

---

#### Prerequisites:

- Python Environment: Ensure you have Python installed on your system. We recommend using Python 3.

- Script Download: Download the [main.py](main.py) script and ensure it's in the same directory as your project.

- Dependencies: Install the required Python packages by running:

>```bash
>pip install -r requirements.txt
>```
>
>or
>
>```bash
>pip install opencv-python
>```

- YOLOv8 Model or Custom Model: Download the YOLOv8 model weights (e.g., 'yolov8.pt') from the [official repository](https://github.com/ultralytics/ultralytics), or use your own model trained on its basis.

- Input Images: Prepare a folder containing the images you want to annotate.

- Configuration & Customization: Customize the script's configuration (e.g., input/output paths, model name) based on your specific setup, including specifying your custom model if applicable.

- Optional GPU Support: If available, consider using a GPU for faster processing. Adjust the device parameter accordingly.

- Label Output Directory: Make sure the directory for storing annotated labels exists or let the script create it.

---

#### Usage
After fulfilling all the prerequisites, run the [main.py](main.py) script, replacing the placeholders (input_folder, output_folder, model_weights, device_type) with your specific values.
The script will automatically detect objects in the images using your YOLOv8 model and generate annotation files in the specified output directory.

---

### License
This project is licensed under the AGPL-3.0 license because it includes modified or borrowed annotation features from Ultralytics, which is also licensed under the AGPL-3.0. Please refer to the [LICENSE](LICENSE) file for more information.
