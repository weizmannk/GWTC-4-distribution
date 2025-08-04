import os
import numpy as np
import pandas as pd
from astropy.table import Table
from bilby.hyper.model import Model
from gwpopulation.models.redshift import PowerLawRedshift
from tqdm import tqdm

from bilby.core.result import read_in_result

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

# ============================================================
# 1. CONFIGURATION
# ============================================================

RESULT_FILE = "data/baseline5_widesigmachi2_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_result.hdf5"
N_SAMPLES = int(1e6)
CHUNK_SIZE = int(1e3)
Z_MAX = 2.3

# ============================================================
# 2. LOAD MAP PARAMETERS FROM BILBY RESULT FILE
# ============================================================

# Load "Broken Power Law + 2 Peaks model" result
result = read_in_result(RESULT_FILE)
posterior = result.posterior.copy()

# Find the sample with maximum likelihood (same as max posterior for uniform priors)
max_likelihood_sample = posterior.loc[np.argmax(posterior.log_likelihood)]

# Set minimum and maximum allowed masses for the model (can be tuned)
max_likelihood_sample["absolute_mmin"] = 0.5
max_likelihood_sample["absolute_mmax"] = 350

print("MAP parameters:")
print(max_likelihood_sample)


# ============================================================
# If we need to replace alpha_1 and alpha_chi  values
# ============================================================
# max_likelihood_sample['alpha_chi'] = 1
# max_likelihood_sample['alpha_1'] = 1


# Use absolute 
exclude_abs = {"alpha_1", "alpha_2", "log_prior"}
for key in max_likelihood_sample.index:
    if key not in exclude_abs:
        max_likelihood_sample[key] = np.abs(max_likelihood_sample[key])

# ============================================================
# 3. CONSTRUCT POPULATION MODEL WITH MAP PARAMETERS
# ============================================================

# Compose the model: mass, spin orientation, spin magnitude, redshift
model = Model([
    matter_matters_pairing,
    iid_spin_orientation_gaussian_isotropic,
    iid_spin_magnitude_beta,
    PowerLawRedshift(z_max=Z_MAX),
])
# Set the parameters from MAP sample
model.parameters.update(max_likelihood_sample)

# ============================================================
# 4. DRAW SYNTHETIC POPULATION SAMPLES (IN CHUNKS)
# ============================================================

n_chunks = int(np.ceil(N_SAMPLES / CHUNK_SIZE))
all_events = []

for i in tqdm(range(n_chunks), desc="Simulating events"):
    current_chunk_size = min(CHUNK_SIZE, N_SAMPLES - i * CHUNK_SIZE)
    events = draw_true_values(
        model=model,
        vt_model=None,
        n_samples=current_chunk_size,
        parameters=None,
    )
    #all_events.append(events)