"""
Implements both independent and mass-ratio paired versions of the 2D mass model, plus the single-mass model.

Authors
-------
- Amanda Farah

"""

from cupy_utils import xp


def matter_matters_primary_secondary_independent(
    dataset, A, NSmin, NSmax, BHmin, BHmax, n0, n1, n2, n3, mbreak, alpha_1, alpha_2
):
    r"""
    Two-dimenstional mass distribution considered in Fishbach, Essick, Holz. Does
    Matter Matter? ApJ Lett 899, 1 (2020) : arXiv:2006.13178 modelling the
    primary and secondary masses as following independent distributions.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' (:math:`m_1`) and
        'mass_ratio' q (:math:`m_2=m_1*q`).
    alpha_1: float
        Powerlaw exponent for compact object below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for compact object above break (:math:`\alpha_2`).
    mbreak: float
        Mass at which the power law exponent switches from alpha_1 to alpha_2.
        Pinned for now to be at BHmin (:math:`\m_{break}`).
    NSmin: float
        Minimum compact object mass (:math:`m_\min`).
    NSmax: float
        Mass at which the notch filter starts (:math:`\gamma_{low}`)
    BHmin: float
        Mass at which the notch filter ends (:math:`\gamma_{high}`)
    BHmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    n{0,1,2,3}: float
        Exponents to set the sharpness of the low mass cutoff, low edge of dip,
        high edge of dip, and high mass cutoff, respectively (:math:`\eta_i`).
    A: float
        depth of the dip between NSmax and BHmin (A).
    """
    print("Colonnes dataset:", dataset.columns)
    p_m1 = matter_matters(
        dataset["mass_1"],
        A,
        NSmin,
        NSmax,
        BHmin,
        BHmax,
        n0,
        n1,
        n2,
        n3,
        mbreak,
        alpha_1,
        alpha_2,
    )
    p_m2 = matter_matters(
        dataset["mass_2"],
        A,
        NSmin,
        NSmax,
        BHmin,
        BHmax,
        n0,
        n1,
        n2,
        n3,
        mbreak,
        alpha_1,
        alpha_2,
    )
    prob = _primary_secondary_general(dataset, p_m1, p_m2)

    # get rid of areas where there are no injections
    prob = xp.where((dataset["mass_1"] > 60) * (dataset["mass_2"] < 3), 0, prob)
    return prob


# def matter_matters_pairing(dataset, A, NSmin, NSmax, BHmin, BHmax, n0, n1, n2, n3, mu1, sig1, mu2, sig2, mbreak, alpha_1, alpha_2, beta_q ):
def matter_matters_pairing(
    dataset,
    A,
    NSmin,
    NSmax,
    BHmin,
    BHmax,
    n0,
    n1,
    n2,
    n3,
    mbreak,
    alpha_1,
    alpha_2,
    beta_q,
):
    r"""
    Two-dimenstional mass distribution considered in Fishbach, Essick, Holz. Does
    Matter Matter? ApJ Lett 899, 1 (2020) : arXiv:2006.13178 modelling the
    primary and secondary masses as following independent distributions.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' (:math:`m_1`) and
        'mass_ratio' q (:math:`m_2=m_1*q`).
    alpha_1: float
        Powerlaw exponent for compact object below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for compact object above break (:math:`\alpha_2`).
    mbreak: float
        Mass at which the power law exponent switches from alpha_1 to alpha_2.
        Pinned for now to be at BHmin (:math:`\m_{break}`).
    NSmin: float
        Minimum compact object mass (:math:`m_\min`).
    NSmax: float
        Mass at which the notch filter starts (:math:`\gamma_{low}`)
    BHmin: float
        Mass at which the notch filter ends (:math:`\gamma_{high}`)
    BHmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    n{0,1,2,3}: float
        Exponents to set the sharpness of the low mass cutoff, low edge of dip,
        high edge of dip, and high mass cutoff, respectively (:math:`\eta_i`).
    A: float
        depth of the dip between NSmax and BHmin (A).
    """

    p_m1 = matter_matters(
        dataset["mass_1"],
        A,
        NSmin,
        NSmax,
        BHmin,
        BHmax,
        n0,
        n1,
        n2,
        n3,
        mbreak,
        alpha_1,
        alpha_2,
    )
    p_m2 = matter_matters(
        dataset["mass_2"],
        A,
        NSmin,
        NSmax,
        BHmin,
        BHmax,
        n0,
        n1,
        n2,
        n3,
        mbreak,
        alpha_1,
        alpha_2,
    )
    prob = _primary_secondary_plaw_pairing(dataset, p_m1, p_m2, beta_q)
    # get rid of areas where there are no injections
    prob = xp.where((dataset["mass_1"] > 60) * (dataset["mass_2"] < 3), 0, prob)
    return prob


def matter_matters(
    mass, A, NSmin, NSmax, BHmin, BHmax, n0, n1, n2, n3, mbreak, alpha_1, alpha_2
):
    r"""
    the single-mass distribution considered in Fishbach, Essick, Holz. Does
    Matter Matter? ApJ Lett 899, 1 (2020) : arXiv:2006.13178

    .. math::
        p(m|\lambda) = n(m|\gamma_{\text{low}}, \gamma_{\text{high}}, A) \times
            l(m|m_{\text{max}}, \eta) \\
                 \times \begin{cases}
                         & m^{\alpha_1} \text{ if } m < \gamma_{\text{low}} \\
                         & m^{\alpha_2} \text{ if } m > \gamma_{\text{low}} \\
                         & 0 \text{ otherwise }
                 \end{cases}.
    
    where $l(m|m_{\text{max}}, \eta)$ is the low pass filter with powerlaw $\eta$
    applied at mass $m_{\text{max}}$,
    $n(m|\gamma_{\text{low}}, \gamma_{\text{high}}, A)$ is the notch
    filter with depth $A$ applied between $\gamma_{\text{low}}$ and 
    $\gamma_{\text{high}}$, and
    $\lambda$ is the subset of hyperparameters $\{ \gamma_{\text{low}},
    \gamma_{\text{high}}, A, \alpha_1, \alpha_2, m_{\text{min}}, m_{\
    text{max}}\}$.
    
    Parameters
    ----------
    mass: array-like
        Mass to evaluate probability at (:math:`m`).
    alpha_1: float
        Powerlaw exponent for compact object below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for compact object above break (:math:`\alpha_2`).
    mbreak: float
        Mass at which the power law exponent switches from alpha_1 to alpha_2.
        Pinned for now to be at BHmin (:math:`\m_{break}`). 
    NSmin: float
        Minimum compact object mass (:math:`m_\min`).
    NSmax: float
        Mass at which the notch filter starts (:math:`\gamma_{low}`)
    BHmin: float
        Mass at which the notch filter ends (:math:`\gamma_{high}`)
    BHmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    n{0,1,2,3}: float
        Exponents to set the sharpness of the low mass cutoff, low edge of dip,
        high edge of dip, and high mass cutoff, respectively (:math:`\eta_i`). 
    A: float
        depth of the dip between NSmax and BHmin (A).
    """
    mbreak = BHmin
    logprob = xp.where(
        (mass >= 1) * (mass <= 100),
        -xp.log(1 + (NSmin / mass) ** n0)
        + xp.log(1.0 - A / ((1 + (NSmax / mass) ** n1) * (1 + (mass / BHmin) ** n2)))
        - xp.log(1 + (mass / BHmax) ** n3)
        + xp.where(mass <= mbreak, alpha_1, alpha_2) * (xp.log(mass) - xp.log(mbreak)),
        -xp.inf,
    )
    return xp.exp(logprob)


def _primary_secondary_general(dataset, p_m1, p_m2):
    return p_m1 * p_m2 * (dataset["mass_1"] >= dataset["mass_2"]) * 2


def _primary_secondary_plaw_pairing(dataset, p_m1, p_m2, beta_pair):
    q = dataset["mass_2"] / dataset["mass_1"]
    return _primary_secondary_general(dataset, p_m1, p_m2) * (q**beta_pair)
