In [1]: run max_posterior_draw.py
/home/weizmann.kiendrebeogo/OBSERVING_SCENARIOS/CBC-populations/GWTC-4-distribution/utils/map_utils.py:21: DeprecationWarning: invalid escape sequence '\o'
  """
A                  0.091462
A2                 0.828165
BHmax            152.055979
BHmin              7.763955
NSmax              4.094744
                    ...
var_99             0.001657
variance           0.984675
xi_spin            0.712869
absolute_mmin      0.500000
absolute_mmax    350.000000
Name: 2411, Length: 334, dtype: float64
A                  0.091462
A2                 0.828165
BHmax            152.055979
BHmin              7.763955
NSmax              4.094744
                    ...
var_99             0.001657
variance           0.984675
xi_spin            0.712869
absolute_mmin      0.500000
absolute_mmax    350.000000
Name: 2411, Length: 334, dtype: float64
Simulating events:   0%|                                                                                                                     | 0/1000 [00:00<?, ?it/s]/home/weizmann.kiendrebeogo/anaconda3/envs/gwpopulation_O4/lib/python3.11/site-packages/bilby/core/prior/interpolated.py:166: DeprecationWarning: `trapz` is deprecated. Use `trapezoid` instead, or one of the numerical integration functions in `scipy.integrate`.
  if np.trapz(self._yy, self.xx) != 1:
/home/weizmann.kiendrebeogo/anaconda3/envs/gwpopulation_O4/lib/python3.11/site-packages/bilby/core/prior/interpolated.py:168: DeprecationWarning: `trapz` is deprecated. Use `trapezoid` instead, or one of the numerical integration functions in `scipy.integrate`.
  self._yy /= np.trapz(self._yy, self.xx)
/home/weizmann.kiendrebeogo/OBSERVING_SCENARIOS/CBC-populations/GWTC-4-distribution/gwpop/mass_models.py:125: RuntimeWarning: overflow encountered in power
  highpass_lower = 1 + (NSmin / mass) ** n0
/home/weizmann.kiendrebeogo/OBSERVING_SCENARIOS/CBC-populations/GWTC-4-distribution/gwpop/mass_models.py:126: RuntimeWarning: overflow encountered in power
  notch_lower = 1.0 - A / ((1 + (NSmax / mass) ** n1) * (1 + (mass / BHmin) ** n2))
/home/weizmann.kiendrebeogo/anaconda3/envs/gwpopulation_O4/lib/python3.11/site-packages/gwpopulation/models/redshift.py:96: DeprecationWarning: `trapz` is deprecated. Use `trapezoid` instead, or one of the numerical integration functions in `scipy.integrate`.
  norm = xp.trapz(normalisation_data, self.zs)
model.prob(data).shape: (10000000,)
02:42 bilby INFO    : Sampling efficiency low. Total samples so far: 0
/home/weizmann.kiendrebeogo/anaconda3/envs/gwpopulation_O4/lib/python3.11/site-packages/gwpopulation/models/redshift.py:96: DeprecationWarning: `trapz` is deprecated. Use `trapezoid` instead, or one of the numerical integration functions in `scipy.integrate`.
  norm = xp.trapz(normalisation_data, self.zs)
/home/weizmann.kiendrebeogo/anaconda3/envs/gwpopulation_O4/lib/python3.11/site-packages/bilby/core/prior/interpolated.py:166: DeprecationWarning: `trapz` is deprecated. Use `trapezoid` instead, or one of the numerical integration functions in `scipy.integrate`.
  if np.trapz(self._yy, self.xx) != 1:
/home/weizmann.kiendrebeogo/anaconda3/envs/gwpopulation_O4/lib/python3.11/site-packages/bilby/core/prior/interpolated.py:168: DeprecationWarning: `trapz` is deprecated. Use `trapezoid` instead, or one of the numerical integration functions in `scipy.integrate`.
  self._yy /= np.trapz(self._yy, self.xx)
model.prob(data).shape: (2,)
Simulating events:   0%|                                                                                                                     | 0/1000 [00:30<?, ?it/s]
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
File ~/OBSERVING_SCENARIOS/CBC-populations/GWTC-4-distribution/max_posterior_draw.py:150
    148 outdir = "O4_result"
    149 os.makedirs(outdir, exist_ok=True)
--> 150 events = sample_max_post(result_file, outdir, n_samples=int(1e6), pdb=True)
    152 if events is not None:
    153     bns_count = (events["mass_1"] < 3).sum()

File ~/OBSERVING_SCENARIOS/CBC-populations/GWTC-4-distribution/max_posterior_draw.py:110, in sample_max_post(result_file, outdir, n_samples, pdb)
    108 for counter in tqdm(range(n_chunks), desc="Simulating events"):
    109     current_chunk_size = min(chunk_size, n_samples - counter * chunk_size)
--> 110     events_group = draw_true_values(
    111         model=model,
    112         vt_model=None,
    113         n_samples=current_chunk_size,
    114         parameters=None,
    115     )
    116     events_group.reset_index(drop=True).to_json(
    117         f"{events_filename}_{counter + 1}.json", indent=4
    118     )
    119     dfs.append(events_group)

File ~/OBSERVING_SCENARIOS/CBC-populations/GWTC-4-distribution/pipe/gwpopulation_pipe_pdb.py:89, in draw_true_values(model, vt_model, n_samples, parameters)
     85 data["mass_2"] = data["mass_1"] * data["mass_ratio"]
     87 print("model.prob(data).shape:", model.prob(data).shape)
---> 89 prob = model.prob(data) * data["mass_1"]
     90 prob *= vt_model(data)
     92 data_df = pd.DataFrame({key: to_numpy(value) for key, value in data.items()})

ValueError: operands could not be broadcast together with shapes (2,) (10000000,)
