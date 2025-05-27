# Examine bifurcation during the optimization in extremely high-dimension space.


# -------- Dataset --------- #

import numpy as np
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST

batch_size = 32

def numpy_collate(batch):
  """
  Collate function specifies how to combine a list of data samples into a batch.
  default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
  """
  return tree_map(np.asarray, default_collate(batch))

def resize_flatten_cast(pic):
  """Convert PIL image to flat (1-dimensional) numpy array."""
  #return np.ravel(np.array(pic, dtype=jnp.float32))
  return np.expand_dims(np.array(pic, dtype=jnp.float32), axis=-1)

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

# Define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=resize_flatten_cast)
# Create pytorch data loader with custom collate function
training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)

# Get the full train dataset (for checking accuracy while training)
train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), 10)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False, transform=resize_flatten_cast)
test_generator = DataLoader(mnist_dataset_test, batch_size=batch_size, collate_fn=numpy_collate)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), 10)


# -------- Model --------- #

from flax import nnx  # The Flax NNX API.
from functools import partial

class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)

  def __call__(self, x):
    x = self.avg_pool(nnx.relu(self.conv1(x)))
    x = self.avg_pool(nnx.relu(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.linear1(x))
    x = self.linear2(x)
    return x

def add_noise(x, scale, random_key):
  std_normal = random.truncated_normal(random.key(random_key), -1, 1, x.shape)
  return x * (1 + scale * std_normal)

class FFD(nnx.Module):
  def __init__(self, *, hidden_units, random_key, noise_scale=0.):
    rngs = nnx.Rngs(random_key)
    self.hidden = nnx.Linear(28*28, hidden_units, rngs=rngs)
    self.output = nnx.Linear(hidden_units, 10, rngs=rngs)
    if noise_scale > 0:
      self.hidden.kernel.value = add_noise(
        self.hidden.kernel.value, noise_scale, random_key)
      self.output.kernel.value = add_noise(
        self.output.kernel.value, noise_scale, random_key)

  def __call__(self, x):
    x = x.reshape(x.shape[0], -1)  # flatten
    return self.output(nnx.relu(self.hidden(x)))

def loss_fn(model: CNN, batch):
  logits = model(batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.
  return loss, logits, grads

@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

import optax
import matplotlib.pyplot as plt

def train(random_key=42, learning_rate=1e-3, eval_every=200, epochs=3,
          noise_scale=0.):
  # model = CNN(rngs=nnx.Rngs(random_key))
  model = FFD(hidden_units=512, random_key=random_key, noise_scale=noise_scale)

  optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
  metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
  )
  metrics_history = {
    'train_loss': [],
    'train_accuracy': [],
    'test_loss': [],
    'test_accuracy': [],
  }

  grads = None
  for epoch in range(epochs):
    for step, (x, y) in enumerate(training_generator):
      _, _, grads = train_step(model, optimizer, metrics, {'image': x, 'label': y})

      if step > 0 and (step % eval_every == 0):  # One training epoch has passed.
        # Log the training metrics.
        for metric, value in metrics.compute().items():  # Compute the metrics.
          metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
        metrics.reset()  # Reset the metrics for the test set.

        # Compute the metrics on the test set after each training epoch.
        for x, y in test_generator:
          eval_step(model, metrics, {'image': x, 'label': y})

        # Log the test metrics.
        for metric, value in metrics.compute().items():
          metrics_history[f'test_{metric}'].append(value)
        metrics.reset()  # Reset the metrics for the next training epoch.
  return model, metrics_history, metrics, grads

#plt.hist(grads.conv2.kernel.value.reshape([-1]), bins=100)
#grads.output.kernel.value
#plt.hist(grads.output.kernel.value.reshape([-1]), bins=100)
#plt.show()

# logfile = 'result.txt'
# with open(logfile, 'w') as f:
#   f.write(repr(model.hidden.kernel.value))
#   f.write('\n\n')
#   f.write(repr(model.hidden.bias.value))
#   f.write('\n\n')
#   f.write(repr(model.output.kernel.value))
#   f.write('\n\n')
#   f.write(repr(model.output.bias.value))
#   f.write('\n\n')
#   f.write(repr(metrics_history['train_accuracy']))

model, metrics_history, metrics, grads = train(noise_scale=0.)
model2, metrics_history2, metrics2, grads2 = train(noise_scale=0.01)

def plot_difference(x, y, precision=1e-2):
  abs_diff = jnp.abs(x - y)
  # Relative difference
  sum_ = jnp.abs(x) + jnp.abs(y)
  denom = jnp.where(sum_ > precision, sum_, precision)
  rel_diff = jnp.abs(x - y) / denom
  # Plot
  fig, axs = plt.subplots(2, 1)
  axs[0].hist(abs_diff.reshape([-1]), bins=100)
  axs[1].hist(rel_diff.reshape([-1]), bins=100)
  plt.show()

plot_difference(model.hidden.kernel.value, model2.hidden.kernel.value)
plot_difference(model.hidden.bias.value, model2.hidden.bias.value)
plot_difference(model.output.kernel.value, model2.output.kernel.value)
plot_difference(model.output.bias.value, model2.output.bias.value)

print(max(metrics_history['train_accuracy']))  # => 0.9446875
print(max(metrics_history2['train_accuracy']))  # => 0.94421875

print(min(metrics_history['train_loss']))  # => 0.25508663
print(min(metrics_history2['train_loss']))  # => 0.2512598
