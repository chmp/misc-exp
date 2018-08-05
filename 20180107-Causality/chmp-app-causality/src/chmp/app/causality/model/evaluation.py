import inspect
import logging

import matplotlib.pyplot as plt
import numpy as np


_logger = logging.getLogger(__name__)


def evaluate(
    est,
    df,
    reward_column="outcome",
    action_column="action",
    value_columns=None,
    propensity_column=None,
    label=None,
):
    action_values = [int(v) for v in np.unique(df[action_column])]

    _logger.info("predict %s", label)
    action, value = est.predict(df)

    _logger.info("run evaluations %s", label)
    return _compute_metrics(
        {
            "size": len(df),
            "label": label,
            "action": action,
            "value": value,
            "action_values": action_values,
            "policy_p": sum(
                (df[action_column] == action_value) * action[:, action_value]
                for action_value in action_values
            ),
            "observed_action": np.asarray(df[action_column], dtype=np.int)
            if action_column is not None
            else None,
            "reward": df[reward_column] if reward_column is not None else None,
            "true_value": np.vstack([df[col] for col in value_columns]).T
            if value_columns is not None
            else None,
            "propensity": df[propensity_column]
            if propensity_column is not None
            else None,
        }
    )


def _compute_metrics(keywords):
    # TODO: add action stats, reward stats, ..
    metrics = {"size": keywords["size"]}

    if keywords["label"] is not None:
        metrics["label"] = keywords["label"]

    # run all evaluation functions
    for func in [
        compute_direct_method_reward,
        compute_mean_observed_reward,
        compute_value_mad,
        compute_value_plots,
        compute_true_reward,
        compute_ips_rewards,
        compute_doubly_robust_reward,
    ]:
        spec = inspect.getfullargspec(func)
        kw = {k: keywords[k] for k in spec.args}

        if any(v is None for v in kw.values()):
            continue

        metrics.update(func(**kw))

    for metric in ["true_reward", "dr_reward", "ips_reward", "dm_reward"]:
        if metric in metrics:
            metrics["best_guess_reward"] = metrics[metric]
            break

    return metrics


def compute_mean_observed_reward(reward):
    return {"mean_observed_reward": np.mean(reward)}


def compute_direct_method_reward(policy_p, reward):
    # WARNING: severely biased
    return {"dm_reward": np.sum(policy_p * reward) / policy_p.sum()}


def compute_value_mad(value, true_value):
    assert value.shape == true_value.shape

    return {
        f"value_mad_{i}": np.mean(np.abs(value[:, i] - true_value[:, i]))
        for i in range(value.shape[1])
    }


def compute_value_plots(value):
    vmin = min(np.min(value[:, 0]), np.min(value[:, 1]))
    vmax = max(np.max(value[:, 0]), np.max(value[:, 1]))

    return {
        "value_plot": canned_hist2d(
            value[:, 0], value[:, 1], range=((vmin, vmax), (vmin, vmax)), bins=(31, 31)
        )
    }


def compute_true_reward(action, true_value):
    true_reward = np.mean(np.sum(action * true_value, axis=1))
    optimal_reward = np.mean(np.max(true_value, axis=1))

    return {
        "true_reward": true_reward,
        "optimal_reward": optimal_reward,
        "regret": true_reward - optimal_reward,
    }


def compute_ips_rewards(reward, policy_p, propensity):
    weight = policy_p / propensity

    return {
        "ips_reward": np.mean(weight * reward),
        "snips_reward": np.mean(weight * reward) / np.mean(weight),
    }


def compute_doubly_robust_reward(
    action, reward, propensity, observed_action, value, action_values, policy_p
):
    observed_value_estimate = value[np.arange(value.shape[0]), observed_action]

    return {
        "dr_reward": (
            np.mean((reward - observed_value_estimate) * policy_p / propensity)
            + sum(
                np.mean(value[:, action_value] * action[:, action_value])
                for action_value in action_values
            )
        )
    }


def canned_hist2d(x, y, **kwargs):
    data, edges_x, edges_y = np.histogram2d(x, y, **kwargs)
    return CannedHist2dPlot(edges_x, edges_y, data)


class CannedHist2dPlot:
    def __init__(self, edges_x, edges_y, data):
        self.edges_x = edges_x
        self.edges_y = edges_y
        self.data = data

    def plot(self, **kwargs):
        plt.pcolor(self.edges_x, self.edges_y, self.data.T, **kwargs)
