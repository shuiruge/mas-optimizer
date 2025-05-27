# Examine bifurcation during the optimization in extremely high-dimension space.


# -------- Dataset --------- #

import numpy as np
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST

batch_size = 32
random_key = 42

def numpy_collate(batch):
  """
  Collate function specifies how to combine a list of data samples into a batch.
  default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
  """
  return tree_map(np.asarray, default_collate(batch))

def resize_flatten_cast(pic):
  """Convert PIL image to flat (1-dimensional) numpy array."""
  return np.expand_dims(np.array(pic, dtype=jnp.float32), axis=-1)

# Define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=resize_flatten_cast)
# Create pytorch data loader with custom collate function
training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)

# Get the full train dataset (for checking accuracy while training)
train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = np.array(mnist_dataset.train_labels)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False, transform=resize_flatten_cast)
test_generator = DataLoader(mnist_dataset_test, batch_size=batch_size, collate_fn=numpy_collate)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = np.array(mnist_dataset_test.test_labels)


# -------- Model --------- #

from flax import nnx  # The Flax NNX API.
from functools import partial
import optax

# class CNN(nnx.Module):
#   """A simple CNN model."""
#
#   def __init__(self, *, rngs: nnx.Rngs):
#     self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
#     self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
#     self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
#     self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
#     self.linear2 = nnx.Linear(256, 10, rngs=rngs)
#
#   def __call__(self, x):
#     x = self.avg_pool(nnx.relu(self.conv1(x)))
#     x = self.avg_pool(nnx.relu(self.conv2(x)))
#     x = x.reshape(x.shape[0], -1)  # flatten
#     x = nnx.relu(self.linear1(x))
#     x = self.linear2(x)
#     return x

class FFD(nnx.Module):
  def __init__(self, *, hidden_units):
    rngs = nnx.Rngs(random_key)
    self.hidden = nnx.Linear(28*28, hidden_units, rngs=rngs)
    self.output = nnx.Linear(hidden_units, 10, rngs=rngs)

  def __call__(self, x):
    x = x.reshape(x.shape[0], -1)  # flatten
    return self.output(nnx.relu(self.hidden(x)))

def _add_noise(x, scale):
  std_normal = random.truncated_normal(random.key(random_key), -1, 1, x.shape)
  return x * (1 + scale * std_normal)

def add_noise(model, noise_scale):
  model.hidden.kernel.value = _add_noise(model.hidden.kernel.value, noise_scale)
  model.output.kernel.value = _add_noise(model.output.kernel.value, noise_scale)

def loss_fn(model, batch):
  logits = model(batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss

@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(model, batch)
  optimizer.update(grads)  # In-place updates.
  return loss

def fit(model, learning_rate=1e-3, eval_every=200, epochs=3):
  optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
  for epoch in range(epochs):
    for step, (x, y) in enumerate(training_generator):
      train_step(model, optimizer, {'image': x, 'label': y})


import matplotlib.pyplot as plt

def relative_diff(x, y, precision):
  sum_ = jnp.abs(x) + jnp.abs(y)
  denom = jnp.where(sum_ > precision, sum_, precision)
  return jnp.abs(x - y) / denom

def plot_difference(x, y, precision=1e-2):
  abs_diff = jnp.abs(x - y)
  # Relative difference
  rel_diff = relative_diff(x, y, precision)
  # Plot
  fig, axs = plt.subplots(2, 1)
  axs[0].hist(abs_diff.reshape([-1]), bins=100)
  axs[1].hist(rel_diff.reshape([-1]), bins=100)
  plt.show()

model = FFD(hidden_units=512)
fit(model)

model2 = FFD(hidden_units=512)
add_noise(model2, 1e-2)
fit(model2)

plot_difference(model.hidden.kernel.value, model2.hidden.kernel.value)
plot_difference(model.hidden.bias.value, model2.hidden.bias.value)
plot_difference(model.output.kernel.value, model2.output.kernel.value)
plot_difference(model.output.bias.value, model2.output.bias.value)

# model3 = nnx.clone(model2)
# model3.output.kernel.value = model3.output.kernel.value * (1 +
#   jnp.where(relative_diff(model.output.kernel.value, model2.output.kernel.value, 1e-2) > 0.5,
#             0., 0.5 * random.normal(random.key(42), model.output.kernel.value.shape))
# )

def training_loss(model):
  return loss_fn(model, {'image': train_images, 'label': train_labels})

def training_accuracy(model):
  predicted_class = jnp.argmax(model(train_images), axis=1)
  return jnp.mean(predicted_class == train_labels)

print(training_loss(model))  # => 0.28564915
print(training_loss(model2))  # => 0.44535416

print(training_accuracy(model))  # => 0.9289167
print(training_accuracy(model2))  # => 0.9127
