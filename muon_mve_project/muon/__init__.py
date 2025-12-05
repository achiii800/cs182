"""
Muon optimizer module.

Provides:
    - MuonSGD: SGD with spectral-norm constraints
    - MuonAdamW: AdamW with spectral-norm constraints
    - Various inner solvers: SpectralClip, FrankWolfe, DualAscent, QuasiNewton, ADMM
    - Metrics: spectral norms, sharpness, GNS, Hessian eigenvalues, subspace drift
"""

from .inner_solvers import (
    BaseInnerSolver,
    SpectralClipSolver,
    FrankWolfeSolver,
    DualAscentSolver,
    QuasiNewtonDualSolver,
    ADMMSolver,
    TangentSpaceProjector,
    get_inner_solver,
    SOLVER_REGISTRY,
)

from .muon_sgd import (
    MuonSGD,
    MuonAdamW,
    create_optimizer,
)

from .metrics import (
    compute_spectral_norms,
    compute_all_singular_values,
    estimate_sharpness,
    estimate_gradient_noise_scale,
    estimate_top_hessian_eigenvalue,
    estimate_hessian_spectrum_lanczos,
    compute_subspace_drift,
    SubspaceDriftTracker,
    lr_stability_scan,
    MetricsLogger,
)

__all__ = [
    # Inner solvers
    "BaseInnerSolver",
    "SpectralClipSolver",
    "FrankWolfeSolver",
    "DualAscentSolver",
    "QuasiNewtonDualSolver",
    "ADMMSolver",
    "TangentSpaceProjector",
    "get_inner_solver",
    "SOLVER_REGISTRY",
    # Optimizers
    "MuonSGD",
    "MuonAdamW",
    "create_optimizer",
    # Metrics
    "compute_spectral_norms",
    "compute_all_singular_values",
    "estimate_sharpness",
    "estimate_gradient_noise_scale",
    "estimate_top_hessian_eigenvalue",
    "estimate_hessian_spectrum_lanczos",
    "compute_subspace_drift",
    "SubspaceDriftTracker",
    "lr_stability_scan",
    "MetricsLogger",
]
