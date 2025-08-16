"""
CBCs population
======================

This script provides an interface to generate gravitational-wave (GW) event samples
using MAP parameters from population inference. It supports the Pairing-Dip-Break (PDB) model
and extends earlier tools to include this functionality.

Authors
-------
- Amanda Farah
- Ramodgwendé Weizmann Kiendrébéogo

Description
-----------
The script extracts MAP hyperparameters from "gwpopulation" result files and draws synthetic events
from a specified population model. It supports both the PDB model and the independent mass model.
Events are saved in chunks to handle large numbers of samples.


Main features:
- Support for Pairing-Dip-Break (PDB) and independent models
- Outputs in JSON and HDF5 formats
- Chunked sampling for handling large event sets
- Clears model cache to avoid memory issues

"""

import logging
import os

import numpy as np
import pandas as pd
from astropy.table import Table
from bilby.core.result import read_in_result
from bilby.hyper.model import Model
from gwpopulation.models.redshift import PowerLawRedshift
from tqdm import tqdm

from gwpop.mass_models import (
    matter_matters_pairing,
    matter_matters_primary_secondary_independent,
)
from gwpop.spin_models import (
    iid_spin_magnitude_gaussian,
    iid_spin_orientation_gaussian_isotropic,
)
from pipe.gwpopulation_pipe_pdb import draw_true_values

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def sample_max_post(result_file, outdir="outdir", n_samples=10, pdb=True):
    """
    Draw samples from the MAP parameters of a population model.

    Parameters
    ----------
    result_file : str
        Path to the Bilby result file (.hdf5).
    n_samples : int
        Total number of samples to generate.
    pdb : bool
        If True, use the PDB model; if False, use the independent model.

    Returns
    -------
    pd.DataFrame
        DataFrame with all simulated events.
    """

    # --- outputs ---
    dirname = os.path.abspath(outdir)
    os.makedirs(dirname, exist_ok=True)
    label = os.path.splitext(os.path.basename(result_file))[0]
    events_filename = os.path.join(dirname, f"{label}_events_baseline5")

    # --- MAP hyperparameters ---
    # Extract the Maximum a Posteriori (MAP) parameters from a Bilby result.
    # - Load the .hdf5 result file and copy the posterior samples.
    # - Compute a score:
    #     * If the prior is non-uniform, use log_likelihood + log_prior (true MAP).
    #     * If the prior is uniform (constant, as in this case), log_prior adds nothing,
    #       this means Maximum Likelihood (ML) and Maximum A Posteriori (MAP) coincide.
    # - Select the sample that maximizes this score.

    # Load "Broken Power Law + 2 Peaks model" result
    result = read_in_result(result_file)

    post = result.posterior.copy()
    if "log_prior" in post and post["log_prior"].nunique() > 1:
        score = post.log_likelihood + post.log_prior
        maxp_samp = post.iloc[np.argmax(score)]
    else:
        maxp_samp = post.iloc[np.argmax(post.log_likelihood)]

    # Set minimum and maximum allowed masses for the model (can be tuned)
    maxp_samp["absolute_mmin"] = 0.5
    maxp_samp["absolute_mmax"] = 350.0

    logger.info(f"[{label}] MAP hyperparameters loaded.")

    # --- model (no caching) ---
    if pdb:
        model = Model(
            [
                matter_matters_pairing,
                iid_spin_orientation_gaussian_isotropic,
                iid_spin_magnitude_gaussian,
                PowerLawRedshift(z_max=2.3),
            ],
            cache=False,
        )
    else:
        model = Model(
            [
                matter_matters_primary_secondary_independent,
                iid_spin_orientation_gaussian_isotropic,
                iid_spin_magnitude_gaussian,
                PowerLawRedshift(z_max=2.3),
            ],
            cache=False,
        )

    model.parameters.update(maxp_samp)

    # rng = np.random.default_rng(seed) if seed is not None else None

    # --- chunked sampling ---
    # We split the total number of samples into smaller "chunks" to avoid
    # memory overload and to save intermediate results to disk.
    dfs = []
    chunk_size = int(1e3)
    n_chunks = int(np.ceil(n_samples / chunk_size))

    # Loop over each chunk
    for counter in tqdm(range(n_chunks), desc="Simulating events"):
        current_chunk_size = min(chunk_size, n_samples - counter * chunk_size)

        # Generate events from the population model
        events_group = draw_true_values(
            model=model, vt_model=None, n_samples=current_chunk_size
        )

        # Save the current chunk immediately as JSON
        # (this prevents memory issues and keeps partial results safe)
        events_group.reset_index(drop=True).to_json(
            f"{events_filename}_{counter + 1}.json", indent=4
        )

        # Keep the chunk in memory for final concatenation
        dfs.append(events_group)

    # --- concatenate all chunks + save global outputs ---
    events = pd.concat(dfs).reset_index(drop=True)
    events.to_json(f"{events_filename}_all.json", indent=4)
    Table.from_pandas(events).write(
        f"{events_filename}_all.h5", path="events", overwrite=True, format="hdf5"
    )

    return events


if __name__ == "__main__":
    result_file = "data/baseline5_widesigmachi2_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_result.hdf5"

    outdir = "O4_result"
    events = sample_max_post(result_file, outdir, n_samples=int(1e6), pdb=True)

    if events is not None:
        bns_count = (events["mass_1"] < 3).sum()
        nsbh_count = ((events["mass_2"] < 3) & (events["mass_1"] >= 3)).sum()
        bbh_count = (events["mass_2"] >= 3).sum()

        print(
            f"Number of BNS: {bns_count}\n"
            f"Number of NSBH: {nsbh_count}\n"
            f"Number of BBH: {bbh_count}"
        )
