import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from absl import app
from absl import flags
from collections import defaultdict
from io import StringIO
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

tf.gfile = tf.io.gfile
utils_ops.tf = tf.compat.v1


FLAGS = flags.FLAGS
flags.DEFINE_string('image_dir', '', 'Path to the input images')
flags.DEFINE_bool('hide_display', False, 'Hide image display')
flags.DEFINE_string('image', '', 'Path to single image')
flags.DEFINE_string('output_dir', '', 'Path to output directory')
FLAGS = flags.FLAGS
FLAGS(sys.argv)

IMAGE_TYPES = ['*.jpg', '*.png']
PATH_TO_LABELS = 'banana_model/banana_detection.pbtxt'


def load_model(model_name):
  model_dir = "banana_graph/saved_model/saved_model.pb"
  model = tf.compat.v2.saved_model.load("banana_graph/saved_model", None)
  model = model.signatures['serving_default']
  return model



category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

if FLAGS.image_dir is not None:
  PATH_TO_TEST_IMAGES_DIR = pathlib.Path(FLAGS.image_dir)
  TEST_IMAGE_PATHS = []
  for imgs in IMAGE_TYPES:
    TEST_IMAGE_PATHS.extend(PATH_TO_TEST_IMAGES_DIR.glob(imgs))
  TEST_IMAGE_PATHS = sorted(list(TEST_IMAGE_PATHS))
  # TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

model_name = 'ssd_mobilenet_coco'
detection_model = load_model(model_name)

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)
  RGB_img = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)

  if FLAGS.output_dir:
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    output_file = os.path.join(FLAGS.output_dir, os.path.basename(image_path))
    print(output_file)
    cv2.imwrite(output_file, RGB_img)

  cv2.imshow('bananas', RGB_img)
  if cv2.waitKey(0) & 0xff == ord('q'):
    cv2.destroyAllWindows()



for image_path in TEST_IMAGE_PATHS:
  show_inference(detection_model, image_path)





