"""
map_utils.py

Utility to extract Maximum a Posteriori (MAP) parameters from Bilby result files.

"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from bilby.core.result import read_in_result


def extract_map_parameters(
    result_file: str,
    parameters: Optional[List[str]] = None,
    sort: bool = True,
    as_series: bool = True,
) -> Union[pd.Series, Dict[str, float]]:
    """
    Extracts Maximum a Posteriori (MAP) parameters from a Bilby result file.

    This function loads a Bilby .hdf5 result file, identifies the posterior sample with the
    highest log-likelihood (the MAP sample), and returns selected population parameters.

    Parameters
    ----------
    result_file : str
        Path to the Bilby result .hdf5 file.
    parameters : list of str, optional
        List of parameter names to extract. If None, uses a default set of common population parameters.
    sort : bool, default=True
        Whether to sort the output by parameter name.
    as_series : bool, default=True
        If True, returns a pandas Series; if False, returns a dictionary.

    Returns
    -------
    pd.Series or dict
        Selected MAP parameter values.

    Notes
    -----
    Units:

    - Masses in solar masses (:math:`M_\odot`)
    - Rates in :math:`\mathrm{Gpc}^{-3}~\mathrm{yr}^{-1}`
    - Hypervolume in :math:`\mathrm{Gpc}^{3}`
    - Spin and beta/alpha parameters are dimensionless


    Example
    -------
    >>> result_file = "data/baseline5_widesigmachi2_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_result.hdf5"
    >>> params = extract_map_parameters(result_file)
    >>> print(params)
    """
    default_parameters = [
        "alpha_1",
        "alpha_2",
        "BHmax",
        "NSmin",
        "NSmax",
        "BHmin",
        "A",
        "n3",
        "beta_q",
        "mu_chi",
        "sigma_chi",
        "sigma_spin",
        "mbreak",
        "n0",
        "n1",
        "n2",
        "amax",
        "alpha_chi",
        "beta_chi",
        "xi_spin",
        "lamb",
        "log_likelihood",
        "log_prior",
        "normalization",
        "selection",
        "pdet_n_effective",
        "surveyed_hypervolume",
        "rate",
        "log_10_rate",
        "min_event_n_effective",
    ]
    if parameters is None:
        parameters = default_parameters

    # Load "Broken Power Law + 2 Peaks model" result
    result = read_in_result(result_file)

    # Find the sample that maximizes the likelihood
    # all priors are uniform so this is the same point that maximizes the posterior
    post = result.posterior.copy()
    maxp = post.loc[np.argmax(post.log_likelihood)]

    maxp["absolute_mmin"] = 0.5
    maxp["absolute_mmax"] = 350
    maxp['alpha_chi'] = 1
    maxp['alpha_1'] = 1
    print(maxp)

    exclude_abs = {"alpha_1", "alpha_2", "log_prior"}
    processed = maxp.copy()

    for key in processed.index:
        if key not in exclude_abs:
            processed[key] = np.abs(processed[key])

    return processed  # pd.Series(processed) if as_series else processed
