import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pickle
import os
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from jax.scipy.stats import norm
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

LATEST_CHECKPOINT = "bnn_latest"

class BNNModel:
    def __init__(self, dataset_path=None, random_seed=456):
        try:
            model = self.load_checkpoint(LATEST_CHECKPOINT)
            self.key = jax.random.key(random_seed)
            self.posterior_samples = model.posterior_samples
            self.pred_dist = model.pred_dist
            self.delta = model.delta
            self.scaled_Xte = model.scaled_Xte
            self.scaled_yte = model.scaled_yte
            self.y_scaler = model.y_scaler
        except:
            self.key = jax.random.key(random_seed)   # initialize empty if no checkpoint
            self.posterior_samples = None
            self.pred_dist = None
            self.delta = 0.3
            self.scaled_Xte = None
            self.scaled_yte = None
            self.y_scaler = None

    @partial(jax.jit, static_argnums=(0,))
    def get_P(self, X):
        predictions = self.pred_dist(rng_key=jax.random.key(1), X=X)["Y"]
        mean = jnp.mean(predictions, axis=0)
        stddev = jnp.std(predictions, axis=0)
        
        P = norm.cdf(self.delta, loc=mean, scale=stddev)
        return jnp.reshape(P, (-1, 1, 1))

    def train(self, dataset_path=None):
        xtr, ytr, xte, yte = self._load_data(dataset_path)
        self._setup_model(xtr, ytr, xte, yte)

    def _load_data(self, dataset_path=None):
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "dataset_2.data")
            
        try:
            total_samples = min(100, 500)  # limit samples like GP 
            all_samples = pd.read_csv(dataset_path, header=None, delimiter=' ').values[:total_samples]
            print(f"Retrieved dataset of {all_samples.shape[0]} samples with {all_samples.shape[1]} dimensions")

            X = all_samples[:, :-1]  
            y = all_samples[:, -1].reshape(-1, 1)
            
            X = np.clip(X, -1e3, 1e3)
            y = np.clip(y, 1e-3, 1e3)
            
            print(f"Data ranges - X: [{X.min():.3f}, {X.max():.3f}], y: [{y.min():.3f}, {y.max():.3f}]")

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=42)

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
        def bnn_model(X, Y=None):
            n_hidden = 20
            w1 = numpyro.sample("w1", dist.Normal(0, 0.5).expand([X.shape[1], n_hidden]))
            b1 = numpyro.sample("b1", dist.Normal(0, 0.5).expand([n_hidden]))
            
            w2 = numpyro.sample("w2", dist.Normal(0, 0.5).expand([n_hidden, 1]))
            b2 = numpyro.sample("b2", dist.Normal(0, 0.5))
            
            hidden = jnp.tanh(jnp.dot(X, w1) + b1)
            mean = jnp.dot(hidden, w2) + b2
            
            sigma = numpyro.sample("sigma", dist.Exponential(1.0))
            numpyro.sample("Y", dist.Normal(mean.squeeze(), sigma), obs=Y)

        print("✨ Training BNN ... ✨")
        kernel = NUTS(bnn_model)
        mcmc = MCMC(kernel, num_warmup=200, num_samples=200, num_chains=1)
        mcmc.run(jax.random.key(0), X=xtr, Y=ytr.squeeze())
        
        self.posterior_samples = mcmc.get_samples()
        self.pred_dist = Predictive(bnn_model, posterior_samples=self.posterior_samples)
        
        print("✨ BNN training completed ✨")
        self.save_checkpoint(LATEST_CHECKPOINT)

    def save_checkpoint(self, f):
        checkpoint = {
            'posterior_samples': self.posterior_samples,
            'delta': self.delta,
            'scaled_Xte': self.scaled_Xte,
            'scaled_yte': self.scaled_yte,
            'y_scaler': self.y_scaler
        }
        with open(f, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved to {f}")

    @classmethod
    def load_checkpoint(cls, f, random_seed=456):
        with open(f, 'rb') as f:
            checkpoint = pickle.load(f)
        
        instance = cls.__new__(cls)
        instance.key = jax.random.key(random_seed)
        instance.posterior_samples = checkpoint['posterior_samples']
        instance.delta = checkpoint['delta']
        instance.scaled_Xte = checkpoint['scaled_Xte']
        instance.scaled_yte = checkpoint['scaled_yte']
        instance.y_scaler = checkpoint['y_scaler']
        
        def bnn_model(X, Y=None):
            n_hidden = 20
            w1 = numpyro.sample("w1", dist.Normal(0, 0.5).expand([X.shape[1], n_hidden]))
            b1 = numpyro.sample("b1", dist.Normal(0, 0.5).expand([n_hidden]))
            w2 = numpyro.sample("w2", dist.Normal(0, 0.5).expand([n_hidden, 1]))
            b2 = numpyro.sample("b2", dist.Normal(0, 0.5))
            hidden = jnp.tanh(jnp.dot(X, w1) + b1)
            mean = jnp.dot(hidden, w2) + b2
            sigma = numpyro.sample("sigma", dist.Exponential(1.0))
            numpyro.sample("Y", dist.Normal(mean.squeeze(), sigma), obs=Y)
        
        instance.pred_dist = Predictive(bnn_model, posterior_samples=instance.posterior_samples) # recreate predictive because can't be pickled
        
        print(f"Checkpoint loaded from {f}")
        return instance

    def evaluate_model(self):
        predictions = self.pred_dist(jax.random.key(42), X=self.scaled_Xte)["Y"]
        pred_mean = jnp.mean(predictions, axis=0)
        pred_std = jnp.std(predictions, axis=0)
       
        mse = mean_squared_error(self.scaled_yte.squeeze(), pred_mean)  # evaluate on scaled data for consistency
        r2 = r2_score(self.scaled_yte.squeeze(), pred_mean)
        
        pred_mean_orig = self.y_scaler.inverse_transform(pred_mean.reshape(-1, 1))
        true_orig = self.y_scaler.inverse_transform(self.scaled_yte.reshape(-1, 1))
        
        return {
            'mse': mse,
            'r2': r2,
            'pred_mean': pred_mean_orig,
            'pred_std': pred_std,
            'true_values': true_orig
        }

if __name__ == '__main__':
    bnn = BNNModel()
    bnn.train()
    
    # print(f"BNN R² = {bnn.evaluate_model()['r2']:.4f}")
    # print(f"BNN MSE = {bnn.evaluate_model()['mse']:.4f}")