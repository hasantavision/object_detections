from PIL import Image
import io
import tensorflow as tf
import cv2

filename = 'path/to/tf.record'
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)

# Define features
read_features = {
    'image/height': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
    'image/width': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
    'image/filename': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/source_id': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/format': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/object/bbox/xmin': tf.io.FixedLenFeature([], dtype=tf.float32, default_value=0.),
    'image/object/bbox/xmax': tf.io.FixedLenFeature([], dtype=tf.float32, default_value=0.),
    'image/object/bbox/ymin': tf.io.FixedLenFeature([], dtype=tf.float32, default_value=0.),
    'image/object/bbox/ymax': tf.io.FixedLenFeature([], dtype=tf.float32, default_value=0.),
    'image/object/class/text': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/object/class/label': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0)
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  try:
    return tf.io.parse_single_example(example_proto, read_features)
  except:
    return None


if __name__ == '__main__':
    for data in raw_dataset:
        a = _parse_function(data)
        if a:
            Image.open(io.BytesIO(a["image/encoded"].numpy())).save(a["image/filename"].numpy().decode())
