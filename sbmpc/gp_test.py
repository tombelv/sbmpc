# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt

# from examples.utils import (
#     clean_legend,
#     use_mpl_style,
# )

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


key = jr.key(123)

# set the default style for plotting
# use_mpl_style()

cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

n = 100
noise = 0.3

key, subkey = jr.split(key)
x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
signal = f(x)
y = signal + jr.normal(subkey, shape=signal.shape) * noise

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)
ytest = f(xtest)

# fig, ax = plt.subplots()
# ax.plot(x, y, "o", label="Observations", color=cols[0])
# ax.plot(xtest, ytest, label="Latent function", color=cols[1])
# ax.legend(loc="best")

kernel = gpx.kernels.RBF()  # 1-dimensional input
meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

print(f"TYPE XTEST = {type(xtest)}")

prior_dist = prior.predict(xtest)

# prior_mean = prior_dist.mean()
# prior_std = prior_dist.variance()
# samples = prior_dist.sample(seed=key, sample_shape=(20,))


# fig, ax = plt.subplots()
# ax.plot(xtest, samples.T, alpha=0.5, color=cols[0], label="Prior samples")
# ax.plot(xtest, prior_mean, color=cols[1], label="Prior mean")
# ax.fill_between(
#     xtest.flatten(),
#     prior_mean - prior_std,
#     prior_mean + prior_std,
#     alpha=0.3,
#     color=cols[1],
#     label="Prior variance",
# )
# ax.legend(loc="best")
# # ax = clean_legend(ax)

# likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

# posterior = prior * likelihood

# opt_posterior, history = gpx.fit_scipy(
#     model=posterior,
#     # we use the negative mll as we are minimising
#     objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
#     train_data=D,
# )

# # print(-gpx.objectives.conjugate_mll(opt_posterior, D))

# latent_dist = opt_posterior.predict(xtest, train_data=D)
# predictive_dist = opt_posterior.likelihood(latent_dist)

# predictive_mean = predictive_dist.mean()
# predictive_std = predictive_dist.stddev()
