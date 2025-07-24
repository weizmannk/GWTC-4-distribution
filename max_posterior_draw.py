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

import os

import numpy as np
import pandas as pd
from astropy.table import Table
from bilby.hyper.model import Model
from gwpopulation.models.redshift import PowerLawRedshift
from tqdm import tqdm

from gwpop.mass_models import (
    matter_matters_pairing,
    matter_matters_primary_secondary_independent,
)
from gwpop.spin_models import (
    iid_spin_magnitude_beta,
    iid_spin_orientation_gaussian_isotropic,
)
from pipe.gwpopulation_pipe_pdb import draw_true_values
from utils.cupy_utils import xp
from utils.map_utils import extract_map_parameters


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
    # Define output filename base
    dirname = os.path.abspath(outdir)
    label = os.path.splitext(os.path.basename(result_file))[0]
    events_filename = os.path.join(dirname, f"{label}_events_baseline5")

    # Extract MAP parameters
    maxp_samp = extract_map_parameters(result_file, as_series=True)

    print(maxp_samp)

    # Select model according to pdb flag
    if pdb:
        model = Model(
            [
                matter_matters_pairing,
                iid_spin_orientation_gaussian_isotropic,
                iid_spin_magnitude_beta,
                PowerLawRedshift(z_max=2.3),
            ]
        )
    else:
        model = Model(
            [
                matter_matters_primary_secondary_independent,
                iid_spin_orientation_gaussian_isotropic,
                iid_spin_magnitude_beta,
                PowerLawRedshift(z_max=2.3),
            ]
        )

    model.parameters.update(maxp_samp)
    # model.parameters.update(dict(beta_q=1.892889))
    # model.parameters.update(dict(lamb=2.7))

    # Sampling in chunks if n_samples is large
    dfs = []
    chunk_size = int(1e3)
    n_chunks = int(np.ceil(n_samples / chunk_size))

    for counter in tqdm(range(n_chunks), desc="Simulating events"):
        current_chunk_size = min(chunk_size, n_samples - counter * chunk_size)
        events_group = draw_true_values(
            model=model,
            vt_model=None,
            n_samples=current_chunk_size,
            parameters=None,
        )
        events_group.reset_index(drop=True).to_json(
            f"{events_filename}_{counter + 1}.json", indent=4
        )
        dfs.append(events_group)

        # Clear Bilby cache to avoid memory issues
        model.parameters.update(dict(A=0.0, lamb=0.1, alpha_chi=2.0, xi_spin=0.2))
        model.prob(
            dict(
                mass_1=xp.array([5, 9]),
                mass_2=xp.array([1, 5]),
                a_1=xp.array([0.5, 0.6]),
                a_2=xp.array([0.5, 0.6]),
                cos_tilt_1=xp.array([0.1, 0.1]),
                cos_tilt_2=xp.array([0.1, 0.1]),
                redshift=xp.array([0.6, 0.6]),
            )
        )
        model.parameters.update(maxp_samp)

    events = pd.concat(dfs).reset_index(drop=True)
    events.to_json(f"{events_filename}_all.json", indent=4)
    Table.from_pandas(events).write(
        f"{events_filename}_all.h5", path="events", overwrite=True, format="hdf5"
    )

    return events


if __name__ == "__main__":
    result_file = "data/baseline5_widesigmachi2_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_result.hdf5"

    outdir = "O4_result"
    os.makedirs(outdir, exist_ok=True)
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
