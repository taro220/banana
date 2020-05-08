This is a banana app.

banana_detection.py requires 

Tensorflow 2.0 Object Detection
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

OpenCV

FLAGS:
--image_dir={Path to image directory}
--output_dir={Path to output directory}
    if given, it will save the images with bounding boxes at this location

--hide_display (not implemented)
    if set, disables the cv2.imshow

--image={image file} (not implemented)
    run detection on a single image




MVP
1. Detect a banana from am image and respond with the image including a box around
the banana.

2. Rate the banana between 1-5 with 1 being completely unripe and 5 being overripe.

Use Fruits 360 Dataset


Datasets from

Horea Muresan, Mihai Oltean, Fruit recognition from images using deep learning, Acta Univ. Sapientiae, Informatica Vol. 10, Issue 1, pp. 26-42, 2018.

Piedad, Eduardo Jr; Ferrer, Laura Vithalie; Pojas, Glydel; Cascabel, Honey Faith; Pantilgan, Rosemarie; Larada, Julaiza; Cabinatan, Ian Paul (2018), “Tier-based Dataset: Musa-Acuminata Banana Fruit Species”, Mendeley Data, v2
http://dx.doi.org/10.17632/zk3tkxndjw.2

Piero Toffanin
https://github.com/pierotofy/dataset_banana