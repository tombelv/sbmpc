from jax import random, jit
import jax.numpy as jnp
import gpjax as gpx
import optax
import numpy as np
import pandas as pd
import os
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from jax.scipy.stats import norm 

from functools import partial

MASS = 0.027
GRAVITY = 9.81
INPUT_HOVER = jnp.array([MASS*GRAVITY, 0., 0., 0.], dtype=jnp.float32)
LATEST_CHECKPOINT = "gp_latest"

num_steps = 50
num_samples = 10
sim_iters = 500
horizon = 25


class GaussianProcessModel(): 
    def __init__(self, dataset_path=None, random_seed=456):
        try:
            model = self.load_checkpoint(LATEST_CHECKPOINT)
            self.key = random.key(random_seed)
            self.training_set = model.training_set
            self.opt_posterior = model.opt_posterior
            self.delta = model.delta
            self.scaled_Xte = model.scaled_Xte
            self.scaled_yte = model.scaled_yte
            self.y_scaler = model.y_scaler
        except:
            self.key = None
            self.training_set = None
            self.opt_posterior = None
            self.delta = 0.3
            self.scaled_Xte = None
            self.scaled_yte = None
            self.y_scaler = None

        self.dataset = dataset_path # some tests provide TODO pass on to train() with correct logic

    @partial(jit, static_argnums=(0,))
    def get_P(self, X):
        latent_dist = self.opt_posterior.predict(X, train_data=self.training_set)

        mean = latent_dist.mean()
        stddev = latent_dist.stddev()

        return norm.cdf(self.delta, loc=mean, scale=stddev)


    def train(self, dataset_path=None):
        xtr, ytr, xte, yte = self._load_data(dataset_path)
        self._setup_model(xtr, ytr, xte, yte)

    def _load_data(self, dataset_path=None):
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "dataset_2.data") # defaut to dataset 2 (I think this is best)
            
        try:
            total_samples = min(100, num_steps * num_samples)  # limit to 100 samples
            all_samples = pd.read_csv(dataset_path, header=None, delimiter=' ').values[:total_samples]
            print(f"Retrieved dataset of {all_samples.shape[0]} samples with {all_samples.shape[1]} dimensions")

            X = all_samples[:, :-1]  
            y = all_samples[:, -1].reshape(-1, 1)
            
            X = np.clip(X, -1e3, 1e3) # handling extreme values
            y = np.clip(y, 1e-3, 1e3)
            
            print(f"Data ranges - X: [{X.min():.3f}, {X.max():.3f}], y: [{y.min():.3f}, {y.max():.3f}]")

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

            x_scaler = StandardScaler().fit(Xtr)
            self.y_scaler = y_scaler = StandardScaler().fit(ytr)
            
            self.scaled_Xte = scaled_Xte = x_scaler.transform(Xte)
            self.scaled_yte = scaled_yte = y_scaler.transform(yte)

            scaled_Xtr = x_scaler.transform(Xtr)
            scaled_ytr = y_scaler.transform(ytr)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")
        
        return scaled_Xtr, scaled_ytr, scaled_Xte, scaled_yte

        
    def _setup_model(self, xtr, ytr, xte, yte):
        n_train, n_covariates = xtr.shape

        kernel = gpx.kernels.Matern52(             
            active_dims=list(range(n_covariates)),
            variance=jnp.var(ytr) * 0.7,
            lengthscale=jnp.ones((n_covariates,)),
            ) + gpx.kernels.RBF(
            active_dims=list(range(n_covariates)),
            variance=jnp.var(ytr) * 0.3,
            lengthscale=0.5 * jnp.ones((n_covariates,)),
            )

        likelihood = gpx.likelihoods.Gaussian(num_datapoints=n_train) 
        mean = gpx.mean_functions.Constant(jnp.mean(ytr))
        prior = gpx.gps.Prior(mean_function = mean, kernel = kernel) 
        posterior = prior * likelihood

        self.training_set = gpx.Dataset(X=xtr, y=ytr)  
        self.test_set = gpx.Dataset(X=xte, y=yte)

        print("✨ Optimising ... ✨")
        self.opt_posterior, history = gpx.fit(
        model=posterior,
        objective=gpx.objectives.conjugate_mll,  
        train_data=self.training_set,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=0.001)
        ),
        num_iters=1500
        )
        
        final_loss = gpx.objectives.conjugate_mll(self.opt_posterior, self.training_set)
        print(f"Final function value: {final_loss:.6f}")

        self.save_checkpoint(LATEST_CHECKPOINT)

    
    def save_checkpoint(self, f):
        checkpoint = {
            'opt_posterior': self.opt_posterior,
            'training_set': self.training_set,
            'test_set': self.test_set,
            'delta': self.delta,
            'scaled_Xte' : self.scaled_Xte, # for evaluate_model() tests
            'scaled_yte' : self.scaled_yte,
            'y_scaler' : self.y_scaler
        }
        with open(f, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved to {f}")


    @classmethod
    def load_checkpoint(cls, f, random_seed=456):
        with open(f, 'rb') as f:
            checkpoint = pickle.load(f)
        
        instance = cls.__new__(cls)
        instance.key = random.key(random_seed)
        instance.opt_posterior = checkpoint['opt_posterior']
        instance.training_set = checkpoint['training_set']
        instance.test_set = checkpoint['test_set']
        instance.delta = checkpoint['delta']
        instance.scaled_Xte = checkpoint['scaled_Xte'] 
        instance.scaled_yte = checkpoint['scaled_yte']
        instance.y_scaler = checkpoint['y_scaler']
        
        print(f"Checkpoint loaded from {f}")
        return instance


    def evaluate_model(self):
        latent_dist = self.opt_posterior.predict(self.scaled_Xte, train_data=self.training_set)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)
        pred_mean = predictive_dist.mean()
        pred_std = jnp.sqrt(predictive_dist.variance())
        
        mse = mean_squared_error(self.scaled_yte, pred_mean) # evaluate on scaled data for consistency
        r2 = r2_score(self.scaled_yte, pred_mean)
        
        pred_mean_orig = self.y_scaler.inverse_transform(pred_mean.reshape(-1, 1))
        true_orig = self.y_scaler.inverse_transform(self.scaled_yte.reshape(-1, 1))
        
        return {
            'mse': mse,
            'r2': r2,
            'pred_mean': pred_mean_orig,
            'pred_std': pred_std,
            'true_values': true_orig
        }    


    def plot_predictions(self, feature_idx=52, save_path="gp_plot.png"):
        latent_dist = self.opt_posterior.predict(self.Xte, train_data=self.training_set)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)
    
        sorted_idx = jnp.argsort(self.Xte[:, feature_idx])
        Xte_sorted = self.Xte[sorted_idx]
        yte_sorted = self.yte[sorted_idx]

        fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
        ax.plot(self.Xtr[:, feature_idx], self.ytr, "x", label="Training Data", color="green", alpha=1)
        ax.plot(Xte_sorted[:, feature_idx], yte_sorted, label="True Values", color="blue", linestyle="solid", linewidth=2)
        ax.legend()
        ax.set(xlabel=f"Input Feature {feature_idx}", ylabel="Output", title="GP Predictions")
        plt.savefig(save_path)
        plt.show()
        


if __name__ == '__main__': # run the file directly to retrain model(s)
    gp = GaussianProcessModel()
    # gp.train()
    print(f"MSE = {gp.evaluate_model()['mse']}")