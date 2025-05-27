from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

# https://mathworld.wolfram.com/SpherePointPicking.html
def random_uniform_on_sphere(key, dim):
    return normalize(random.normal(key, dim))

def random_direction(key, dim):
    return jnp.abs(random_uniform_on_sphere(key, dim))

def norm(x):
    return jnp.sqrt(jnp.sum(jnp.square(x)))

def normalize(x, eps=0.):
    return x / (norm(x) + eps)

dim = 100000
#dim = 10
random_key = 42
experiments = 15

keys = random.split(random.key(random_key), experiments)
directions = [normalize(jnp.ones(dim))] + [random_direction(k, dim) for k in keys]
inner_product_mat = jnp.zeros([len(directions), len(directions)])
for i in range(len(directions)):
    for j in range(i, len(directions)):
        inner_product = jnp.dot(directions[i], directions[j])
        inner_product_mat = inner_product_mat.at[i, j].set(inner_product)
        inner_product_mat = inner_product_mat.at[j, i].set(inner_product)

plt.matshow(inner_product_mat)
plt.colorbar()
plt.show()

inner_product_mat
