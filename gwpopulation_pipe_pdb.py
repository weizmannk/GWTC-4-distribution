"""
gwpopulation_pipe_pdb
=====================

This module extends the original ``gwpopulation_pipe`` by incorporating support for the **PDB model parameters**.

Authors
-------
- Amanda Farah
- Ramodgwendé Weizmann Kiendrébéogo

Description
-----------
Provides tools for generating synthetic gravitational-wave (GW) event samples from population models.
It supports custom selection functions and explicit parameter bounds, making it suitable for GW population inference studies.

Functions
---------
- draw_true_values: Draws synthetic GW event parameters from a specified population model.
- _draw_from_prior: Generates samples uniformly from parameter prior distributions.

"""

import numpy as np
import pandas as pd
from bilby.core.utils import logger
from bilby.gw.cosmology import get_cosmology
from bilby.gw.prior import UniformSourceFrame

from cupy_utils import to_numpy, xp

# Parameter bounds
BOUNDS = {
    "mass_1": (1, 100),
    "mass_ratio": (0, 1),
    "a_1": (0, 1),
    "a_2": (0, 1),
    "cos_tilt_1": (-1, 1),
    "cos_tilt_2": (-1, 1),
    "redshift": (0, 1.5),
}

cosmology = get_cosmology()


def draw_true_values(model, vt_model=None, n_samples=40, parameters=None):
    """
    Draw synthetic gravitational-wave event parameters using rejection sampling from a population model.

    Parameters
    ----------
    model : bilby.hyper.model.Model
        The gravitational-wave population model to sample from.
    vt_model : gwpopulation.vt.GridVT or callable, optional
        Selection function (VT) model that returns a detection probability for each event.
        If not provided, a flat selection function (all events equally detectable) is assumed.
    n_samples : int, optional
        Number of synthetic samples to generate. Default is 40.
    parameters : dict, optional
        Dictionary of parameters to restore the model's original state after sampling.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing `n_samples` synthetic gravitational-wave events.

    Notes
    -----
    - The rejection sampling approach may be inefficient for sharply peaked distributions.
    - When a selection function (`vt_model`) is provided, it is applied to weight event detectability.
    """

    if vt_model is None:

        def vt_model(x):
            return 1
    else:
        raise NotImplementedError("Custom VT models are not yet implemented.")

    total_samples = pd.DataFrame()
    n_per_iteration = n_samples * 10000

    while True:
        data = _draw_from_prior(n_samples=n_per_iteration)
        data["mass_2"] = data["mass_1"] * data["mass_ratio"]

        print("model.prob(data).shape:", model.prob(data).shape)

        prob = model.prob(data) * data["mass_1"]
        prob *= vt_model(data)

        data_df = pd.DataFrame({key: to_numpy(value) for key, value in data.items()})
        data_df["prob"] = to_numpy(prob)
        total_samples = pd.concat([total_samples, data_df], ignore_index=True)

        max_prob = total_samples["prob"].max()
        total_samples = total_samples[
            total_samples["prob"] > max_prob / n_per_iteration
        ]

        keep = total_samples["prob"].values >= np.random.uniform(
            0, max_prob, len(total_samples)
        )

        if keep.sum() >= n_samples:
            total_samples = total_samples[keep].iloc[:n_samples]
            break

        logger.info(
            f"Sampling efficiency low. Total samples so far: {len(total_samples)}"
        )

        # Clear model cache workaround
        model.parameters.update(dict(A=0.0, lamb=0.1, alpha_chi=2.0, xi_spin=0.2))
        model.prob(
            {
                "mass_1": xp.array([5, 9]),
                "mass_2": xp.array([1, 5]),
                "a_1": xp.array([0.5, 0.6]),
                "a_2": xp.array([0.5, 0.6]),
                "cos_tilt_1": xp.array([0.1, 0.1]),
                "cos_tilt_2": xp.array([0.1, 0.1]),
                "redshift": xp.array([0.6, 0.6]),
            }
        )

        if parameters is not None:
            model.parameters.update(parameters)

    total_samples = total_samples.drop(columns="prob")
    return total_samples


def _draw_from_prior(n_samples):
    """
    Draws samples uniformly from prior bounds for each parameter.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.


    Returns
    -------
    dict
        Dictionary containing arrays of sampled parameters.
    """
    samples = {
        key: xp.random.uniform(low, high, n_samples)
        for key, (low, high) in BOUNDS.items()
    }

    samples["redshift"] = xp.asarray(
        UniformSourceFrame(
            minimum=BOUNDS["redshift"][0],
            maximum=BOUNDS["redshift"][1],
            name="redshift",
        ).sample(n_samples)
    )

    return samples
