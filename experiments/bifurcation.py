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

def reshape_cast(pic):
  """Output shape: [None, 28, 28, 1]."""
  return np.expand_dims(np.array(pic, dtype=jnp.float32), axis=-1)

# Define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=reshape_cast)
# Create pytorch data loader with custom collate function
training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)
X_train = np.array(mnist_dataset.train_data)
y_train = np.array(mnist_dataset.train_labels)


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
    self.hidden_layer = nnx.Linear(28*28, hidden_units, rngs=rngs)
    self.output_layer = nnx.Linear(hidden_units, 10, rngs=rngs)

  def __call__(self, x):
    x = x.reshape(x.shape[0], -1)  # flatten
    return self.output_layer(nnx.relu(self.hidden_layer(x)))

def loss_fn(model, X, y):
  logits = model(X)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=y
  ).mean()
  return loss

@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, X, y):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(model, X, y)
  optimizer.update(grads)  # In-place updates.
  return loss

def fit(model, learning_rate=1e-3, eval_every=200, epochs=3):
  optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
  for epoch in range(epochs):
    for step, (X, y) in enumerate(training_generator):
      train_step(model, optimizer, X, y)

def training_loss(model):
  return loss_fn(model, X_train, y_train)

def training_accuracy(model):
  predicted_class = jnp.argmax(model(X_train), axis=1)
  return jnp.mean(predicted_class == y_train)


# -------- Bifurcation --------- #

import matplotlib.pyplot as plt

def _add_noise(x, scale):
  std_normal = random.truncated_normal(random.key(random_key), -1, 1, x.shape)
  return x * (1 + scale * std_normal)

def add_noise(model, noise_scale):
  model.hidden_layer.kernel.value = _add_noise(model.hidden_layer.kernel.value, noise_scale)
  model.output_layer.kernel.value = _add_noise(model.output_layer.kernel.value, noise_scale)

def relative_diff(x, y, precision):
  sum_ = jnp.abs(x) + jnp.abs(y)
  denom = jnp.where(sum_ > precision, sum_, precision)
  return jnp.abs(x - y) / denom

def plot_difference(x, y, precision=1e-2):
  abs_diff = jnp.abs(x - y)
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

plot_difference(model.hidden_layer.kernel.value, model2.hidden_layer.kernel.value)
plot_difference(model.hidden_layer.bias.value, model2.hidden_layer.bias.value)
plot_difference(model.output_layer.kernel.value, model2.output_layer.kernel.value)
plot_difference(model.output_layer.bias.value, model2.output_layer.bias.value)

# model3 = nnx.clone(model2)
# model3.output_layer.kernel.value = model3.output_layer.kernel.value * (1 +
#   jnp.where(relative_diff(model.output_layer.kernel.value, model2.output_layer.kernel.value, 1e-2) > 0.5,
#             0., 0.5 * random.normal(random.key(42), model.output_layer.kernel.value.shape))
# )

print(training_loss(model))  # => 0.28564915
print(training_loss(model2))  # => 0.44535416

print(training_accuracy(model))  # => 0.9289167
print(training_accuracy(model2))  # => 0.9127
