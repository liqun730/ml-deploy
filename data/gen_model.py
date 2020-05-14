import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# loading training data
BATCH_SIZE = 32
IMAGE_RES = 224

def format_image(image, label):
    # hub image modules exepct data normalized to [0,1]
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label

(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:85%]', 'train[85%:]'],
    with_info=True,
    as_supervised=True,
)
num_examples = info.splits['train'].num_examples
train_batches = train_examples.cache() \
    .shuffle(num_examples//4) \
    .map(format_image) \
    .batch(BATCH_SIZE) \
    .prefetch(1)
validation_batches = validation_examples.cache() \
    .map(format_image) \
    .batch(BATCH_SIZE) \
    .prefetch(1)

# transfer learning: using Google mobilenet as pre-trained model
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES,3))
feature_extractor.trainable = False

# add a final layer for binary classification
model = tf.keras.Sequential([feature_extractor,layers.Dense(2, activation='sigmoid')])

# fit model
model.compile(
    optimizer='adam', 
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

classifier = model.fit(train_batches, epochs=5, validation_data=validation_batches)

# save model
model.save('classfier.h5')
