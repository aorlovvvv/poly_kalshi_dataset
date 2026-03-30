"""
Reinforcement Learning (PPO) Agent for Prediction Market Position Sizing.

Implements a PPO agent that learns optimal position sizing given microstructure
features. Designed to be used within the autoresearch framework as an alternative
to the logistic regression strategy.

State: (price, rolling_vol_10, taker_buy, lag_ret_1, lag_ret_2,
        current_position, unrealized_pnl, drawdown)
Action: Continuous position size in [-1, 1]
Reward: PnL minus transaction costs, with optional Sharpe shaping

References:
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms."
- PPO Sharpe 1.75 vs DQN 1.12 in optimal execution (market making literature)

Usage:
    # As a strategy within autoresearch:
    from rl_agent import PPOStrategy
    strategy = PPOStrategy()
    strategy.train(X_train, y_train, feature_names)

    # Standalone training:
    python -m src.autoresearch.rl_agent --episodes 100
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import from the fixed harness — try relative first (package mode),
# fall back to bare import (running from src/autoresearch/ directory).
try:
    from .prepare import Strategy, PortfolioState, FEATURE_COLUMNS
except ImportError:
    from prepare import Strategy, PortfolioState, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Trading Environment
# ---------------------------------------------------------------------------

class PredictionMarketEnv:
    """
    Gym-like environment for prediction market trading.

    Observation: feature vector + portfolio state
    Action: desired position in [-1, 1]
    Reward: realized PnL minus transaction cost
    """

    def __init__(self, features: np.ndarray, next_rets: np.ndarray,
                 cost_per_contract: float = 0.07, reward_shaping: str = "pnl"):
        self.features = features
        self.next_rets = next_rets
        self.n_steps = len(features)
        self.cost_per_unit = cost_per_contract
        self.reward_shaping = reward_shaping

        # State
        self.step_idx = 0
        self.position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.pnl_history = []

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.step_idx = 0
        self.position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.pnl_history = []
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Get current observation: features + portfolio state."""
        if self.step_idx >= self.n_steps:
            return np.zeros(self.features.shape[1] + 3)

        feat = self.features[self.step_idx]
        portfolio = np.array([
            self.position,
            self.equity - 1.0,  # unrealized PnL (relative to initial)
            max(0, 1.0 - self.equity / self.peak_equity),  # drawdown
        ])
        return np.concatenate([feat, portfolio])

    def step(self, action: float) -> tuple:
        """
        Execute one trading step.

        Args:
            action: Desired position in [-1, 1]

        Returns:
            (observation, reward, done, info)
        """
        action = np.clip(action, -1.0, 1.0)

        if self.step_idx >= self.n_steps - 1:
            return self._get_obs(), 0.0, True, {}

        # Trade
        next_ret = self.next_rets[self.step_idx]
        if np.isnan(next_ret):
            next_ret = 0.0

        trade_delta = abs(action - self.position)
        cost = trade_delta * self.cost_per_unit
        pnl = action * next_ret - cost

        # Update state
        self.position = action
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.pnl_history.append(pnl)

        # Reward shaping
        if self.reward_shaping == "pnl":
            reward = pnl
        elif self.reward_shaping == "sharpe":
            # Rolling Sharpe reward: encourages risk-adjusted returns
            if len(self.pnl_history) > 20:
                recent = np.array(self.pnl_history[-20:])
                std = np.std(recent)
                reward = pnl / (std + 1e-8)
            else:
                reward = pnl
        elif self.reward_shaping == "sortino":
            if len(self.pnl_history) > 20:
                recent = np.array(self.pnl_history[-20:])
                downside = recent[recent < 0]
                downside_std = np.std(downside) if len(downside) > 1 else 1e-4
                reward = pnl / (downside_std + 1e-8)
            else:
                reward = pnl
        else:
            reward = pnl

        self.step_idx += 1
        done = self.step_idx >= self.n_steps - 1
        obs = self._get_obs()

        return obs, reward, done, {"pnl": pnl, "equity": self.equity}

    @property
    def obs_dim(self) -> int:
        return self.features.shape[1] + 3

    @property
    def action_dim(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# PPO Network (Actor-Critic)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """Actor-Critic network for PPO."""

        def __init__(self, obs_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            # Actor: outputs mean of action distribution
            self.actor_mean = nn.Linear(hidden_dim, 1)
            self.actor_log_std = nn.Parameter(torch.zeros(1))
            # Critic: outputs state value
            self.critic = nn.Linear(hidden_dim, 1)

        def forward(self, obs):
            shared = self.shared(obs)
            action_mean = torch.tanh(self.actor_mean(shared))  # Bound to [-1, 1]
            action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
            value = self.critic(shared)
            return action_mean, action_std, value

        def get_action(self, obs):
            """Sample action from policy."""
            action_mean, action_std, value = self.forward(obs)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return (torch.tanh(action).detach().numpy().flatten()[0],
                    log_prob.detach(), value.detach())

        def evaluate(self, obs, action):
            """Evaluate log probability and value of given actions."""
            action_mean, action_std, value = self.forward(obs)
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return log_prob, value, entropy


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Proximal Policy Optimization trainer."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64,
                 lr: float = 3e-4, gamma: float = 0.99, eps_clip: float = 0.2,
                 epochs_per_update: int = 4, batch_size: int = 64):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PPO. Install with: pip install torch")

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size

        self.model = ActorCritic(obs_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def compute_returns(self, rewards: list, dones: list, values: list) -> list:
        """Compute discounted returns with GAE."""
        returns = []
        gae = 0
        next_value = 0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0
                gae = 0
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * 0.95 * gae  # GAE lambda = 0.95
            returns.insert(0, gae + values[i])
            next_value = values[i]

        return returns

    def update(self, buffer: dict):
        """PPO update step."""
        obs = torch.FloatTensor(np.array(buffer["obs"]))
        actions = torch.FloatTensor(np.array(buffer["actions"])).unsqueeze(1)
        old_log_probs = torch.stack(buffer["log_probs"])
        returns = torch.FloatTensor(buffer["returns"]).unsqueeze(1)
        values = torch.FloatTensor(buffer["values"]).unsqueeze(1)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs_per_update):
            # Mini-batch updates
            indices = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                log_probs, curr_values, entropy = self.model.evaluate(
                    obs[idx], actions[idx])

                ratios = torch.exp(log_probs - old_log_probs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                    1 + self.eps_clip) * advantages[idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (returns[idx] - curr_values).pow(2).mean()
                entropy_loss = -0.01 * entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def train_on_env(self, env: PredictionMarketEnv, n_episodes: int = 50,
                     max_steps: int = 10000, verbose: bool = True) -> list:
        """Train PPO agent on the trading environment."""
        episode_returns = []

        for ep in range(n_episodes):
            obs = env.reset()
            buffer = {"obs": [], "actions": [], "log_probs": [],
                      "rewards": [], "dones": [], "values": []}

            total_reward = 0
            for step in range(min(max_steps, env.n_steps - 1)):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, log_prob, value = self.model.get_action(obs_tensor)

                next_obs, reward, done, info = env.step(action)

                buffer["obs"].append(obs)
                buffer["actions"].append(action)
                buffer["log_probs"].append(log_prob)
                buffer["rewards"].append(reward)
                buffer["dones"].append(done)
                buffer["values"].append(value.item())

                total_reward += reward
                obs = next_obs

                if done:
                    break

            # Compute returns and update
            buffer["returns"] = self.compute_returns(
                buffer["rewards"], buffer["dones"], buffer["values"])
            self.update(buffer)

            episode_returns.append(total_reward)
            if verbose and (ep + 1) % 10 == 0:
                mean_ret = np.mean(episode_returns[-10:])
                print(f"  Episode {ep+1}/{n_episodes}: "
                      f"return={total_reward:.4f}, "
                      f"mean_10={mean_ret:.4f}, "
                      f"equity={env.equity:.4f}")

        return episode_returns


# ---------------------------------------------------------------------------
# PPO Strategy (conforms to autoresearch Strategy interface)
# ---------------------------------------------------------------------------

class PPOStrategy(Strategy):
    """
    PPO-based trading strategy that conforms to the autoresearch interface.

    This can be used as a drop-in replacement for the logistic regression
    strategy in train.py.
    """

    def __init__(self, hidden_dim: int = 64, n_episodes: int = 50,
                 lr: float = 3e-4, reward_shaping: str = "sharpe"):
        self.hidden_dim = hidden_dim
        self.n_episodes = n_episodes
        self.lr = lr
        self.reward_shaping = reward_shaping
        self.trainer = None
        self._feature_names = []
        self._feature_importances = {}

    def name(self) -> str:
        return f"ppo-{self.hidden_dim}h-{self.n_episodes}ep-{self.reward_shaping}"

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: list[str]) -> None:
        """Train PPO agent on training data."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PPO strategy")

        self._feature_names = feature_names

        # We need next_ret for the environment, but it's not in features.
        # The harness provides this separately in evaluate(). For training,
        # we use y_train (binary tail labels) to construct synthetic returns.
        # This is a simplification — in practice, we'd want the actual next_ret.
        # For the autoresearch loop, prepare.py handles this correctly.

        # Construct approximate returns from features
        # Use 'ret' feature if available, otherwise use random walk
        ret_idx = feature_names.index("ret") if "ret" in feature_names else -1

        if ret_idx >= 0:
            # Shift ret by 1 to get approximate next_ret
            next_rets = np.roll(X_train[:, ret_idx], -1)
            next_rets[-1] = 0
        else:
            # Fallback: tiny random returns
            next_rets = np.random.randn(len(X_train)) * 0.001

        # Create environment
        env = PredictionMarketEnv(
            features=X_train,
            next_rets=next_rets,
            cost_per_contract=0.07,
            reward_shaping=self.reward_shaping
        )

        # Train PPO
        obs_dim = env.obs_dim
        self.trainer = PPOTrainer(
            obs_dim=obs_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr
        )

        print(f"Training PPO agent: {self.n_episodes} episodes, "
              f"obs_dim={obs_dim}, hidden={self.hidden_dim}")
        self.trainer.train_on_env(env, n_episodes=self.n_episodes, verbose=True)

        # Compute approximate feature importances via gradient saliency
        self._compute_feature_importances(X_train)

    def _compute_feature_importances(self, X: np.ndarray):
        """Estimate feature importances via input gradient magnitude."""
        if not TORCH_AVAILABLE or self.trainer is None:
            self._feature_importances = {f: 1.0/len(self._feature_names)
                                          for f in self._feature_names}
            return

        # Sample subset for efficiency
        n_sample = min(1000, len(X))
        idx = np.random.choice(len(X), n_sample, replace=False)
        X_sample = X[idx]

        # Add portfolio state columns
        portfolio = np.zeros((n_sample, 3))
        obs = np.hstack([X_sample, portfolio])
        obs_tensor = torch.FloatTensor(obs)
        obs_tensor.requires_grad_(True)

        # Forward pass
        action_mean, _, _ = self.trainer.model(obs_tensor)
        action_mean.sum().backward()

        # Gradient magnitude per feature
        grads = obs_tensor.grad.numpy()[:, :len(self._feature_names)]
        importances = np.mean(np.abs(grads), axis=0)
        total = importances.sum()
        if total > 0:
            importances = importances / total

        self._feature_importances = {
            name: round(float(imp), 6)
            for name, imp in zip(self._feature_names, importances)
        }

    def predict_tail_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Predict tail probability.

        For PPO, we use the absolute value of the action as a proxy:
        larger actions indicate the agent sees opportunity (which correlates
        with high-volatility / tail-event regimes).
        """
        if not TORCH_AVAILABLE or self.trainer is None:
            return np.full(len(X), 0.5)

        # Add portfolio state
        portfolio = np.zeros((len(X), 3))
        obs = np.hstack([X, portfolio])
        obs_tensor = torch.FloatTensor(obs)

        with torch.no_grad():
            action_mean, _, _ = self.trainer.model(obs_tensor)
            probs = torch.abs(action_mean).numpy().flatten()

        return np.clip(probs, 0.0, 1.0)

    def size_position(self, tail_prob: float, state: PortfolioState) -> float:
        """
        Size position using the PPO policy.

        Uses the trained policy to determine optimal position size.
        """
        if not TORCH_AVAILABLE or self.trainer is None:
            return 0.0

        # Construct observation (we don't have full features here,
        # just tail_prob, so use a simplified approach)
        # In practice, the autoresearch harness calls predict_tail_probability
        # in batch mode, so this is only used for online/incremental scenarios.
        return float(np.clip(2.0 * tail_prob - 1.0, -1.0, 1.0))

    def get_feature_importances(self) -> dict[str, float]:
        return self._feature_importances


# ---------------------------------------------------------------------------
# Standalone training script
# ---------------------------------------------------------------------------

def main():
    """Train PPO agent standalone and report metrics."""
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO agent for prediction markets")
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--reward", type=str, default="sharpe",
                        choices=["pnl", "sharpe", "sortino"])
    args = parser.parse_args()

    try:
        from .prepare import load_data, evaluate
    except ImportError:
        from prepare import load_data, evaluate

    print("Loading data...")
    train_df, test_df, feature_names = load_data()

    X_train = train_df[feature_names].values
    y_train = train_df["target"].values
    X_test = test_df[feature_names].values
    y_test = test_df["target"].values
    next_rets = test_df["next_ret"].values

    strategy = PPOStrategy(
        hidden_dim=args.hidden,
        n_episodes=args.episodes,
        reward_shaping=args.reward
    )

    print("\nTraining PPO agent...")
    strategy.train(X_train, y_train, feature_names)

    print("\nEvaluating...")
    metrics = evaluate(strategy, X_test, y_test, next_rets, test_df)

    print(f"\n--- PPO Results ---")
    print(f"Strategy: {strategy.name()}")
    for k, v in metrics.items():
        if k != "feature_importances":
            print(f"  {k}: {v}")
    print(f"\nTop features:")
    fi = metrics["feature_importances"]
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat:25s} {imp:.4f}")


if __name__ == "__main__":
    main()
