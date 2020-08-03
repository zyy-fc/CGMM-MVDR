#!/usr/bin/env python

# wujian@2018
"""
Trainer for some spatial clustering algorithm

CGMM Trainer
Reference:
    Higuchi T, Ito N, Yoshioka T, et al. Robust MVDR beamforming using time-frequency masks
    for online/offline ASR in noise[C]//Acoustics, Speech and Signal Processing (ICASSP),
    2016 IEEE International Conference on. IEEE, 2016: 5210-5214.

CACGMM Trainer
Reference:
    N. Ito, S. Araki, and T. Nakatani, “Complex angular central Gaussian mixture model for 
    directional statistics in mask-based microphone array signal processing,” in European 
    Signal Processing Conference (EUSIPCO). IEEE, 2016, pp. 1153–1157.
"""
import pickle
import numpy as np

from scipy.optimize import linear_sum_assignment
import scipy.io as scio
from .utils import get_logger, EPSILON
import copy

logger = get_logger(__name__)

supported_plan = {
    257: [[20, 70, 170], [2, 90, 190], [2, 50, 150], [2, 110, 210],
          [2, 30, 130], [2, 130, 230], [2, 0, 110], [2, 150, 257]],
    513: [[20, 100, 200], [2, 120, 220], [2, 80, 180], [2, 140, 240],
          [2, 60, 160], [2, 160, 260], [2, 40, 140], [2, 180, 280],
          [2, 0, 120], [2, 200, 300], [2, 220, 320], [2, 240, 340],
          [2, 260, 360], [2, 280, 380], [2, 300, 400], [2, 320, 420],
          [2, 340, 440], [2, 360, 460], [2, 380, 480], [2, 400, 513]]
}

beta  = 1e-6

def norm_observation(mat, axis=-1, eps=EPSILON):
    """
    L2 normalization for observation vectors
    """
    denorm = np.linalg.norm(mat, axis=axis, keepdims=True)
    denorm = np.maximum(denorm, eps)
    return mat / denorm


def permu_aligner(masks, transpose=False):
    """
    Solve permutation problems for clustering based mask algorithm
    Reference: "https://raw.githubusercontent.com/fgnt/pb_bss/master/pb_bss/permutation_alignment.py"
    args:
        masks: K x T x F
    return:
        aligned_masks: K x T x F
    """
    if masks.ndim != 3:
        raise RuntimeError("Expect 3D TF-masks, K x T x F or K x F x T")
    if transpose:
        masks = np.transpose(masks, (0, 2, 1))
    K, _, F = masks.shape
    # normalized masks, for cos distance, K x T x F
    feature = norm_observation(masks, axis=1)
    # K x F
    mapping = np.stack([np.ones(F, dtype=np.int) * k for k in range(K)])

    if F not in supported_plan:
        raise ValueError(f"Unsupported num_bins: {F}")
    for itr, beg, end in supported_plan[F]:
        for _ in range(itr):
            # normalized centroid, K x T
            centroid = np.mean(feature[..., beg:end], axis=-1)
            centroid = norm_observation(centroid, axis=-1)
            go_on = False
            for f in range(beg, end):
                # K x K
                score = centroid @ norm_observation(feature[..., f], axis=-1).T
                # derive permutation based on score matrix
                index, permu = linear_sum_assignment(score, maximize=True)
                # not ordered
                if np.sum(permu != index) != 0:
                    feature[..., f] = feature[permu, :, f]
                    mapping[..., f] = mapping[permu, f]
                    go_on = True
            if not go_on:
                break
    # K x T x F
    permu_masks = np.zeros_like(masks)
    for f in range(F):
        permu_masks[..., f] = masks[mapping[..., f], :, f]
    return permu_masks


class Covariance(object):
    """
    Object of covariance matrix
    """
    def __init__(self, covar, force_hermitian=True):
        if force_hermitian:
            covar_h = np.einsum("...xy->...yx", covar.conj())
            covar = (covar + covar_h) / 2
        self.covar = covar
        self.M = self.covar.shape[-1]

    def mat(self, inv=False):
        """
        Return R or R^{-1}
        """
        # K x F x M x M
        if not inv:
            return self.covar
        else:
            return np.linalg.inv(self.covar +
                                 beta * np.eye(self.M, self.M, dtype=np.complex))

    def det(self, log=True):
        """
        Return (log) det of R
        """
        # K x F x M => K x F x 1
        if log:
            covar_det = np.log(np.linalg.det(self.covar + beta * np.eye(self.M,
                                                            self.M, dtype=np.complex)))
            return np.expand_dims(covar_det, axis=2)

        else:
            return np.expand_dims(np.linalg.det(self.covar + beta * np.eye(self.M,
                                                    self.M, dtype=np.complex)), axis=2)


class Distribution(object):
    """
    Basic distribution class
    """
    def __init__(self, covar=None):
        self.parameters = {
            "covar": None if covar is None else Covariance(covar)
        }

    def check_status(self):
        """
        Check if distribution is initialized
        """
        for key, value in self.parameters.items():
            if value is None:
                raise RuntimeError(
                    f"{key} is not initialized in the distribution")

    def update_covar(self, covar, force_hermitian=True):
        """
        Update covariance matrix (K x F x M x M)
        """
        self.parameters["covar"] = Covariance(covar,
                                              force_hermitian=force_hermitian)

    def covar(self, inv=False):
        """
        Return R or R^{-1}
        """
        # K x F x M x M
        return self.parameters["covar"].mat(inv=inv)

    def log_pdf(self, obs, **kwargs):
        """
        Return value of log-pdf
        """
        raise NotImplementedError

    def pdf(self, obs, **kwargs):
        """
        Return value of pdf
        """
        raise NotImplementedError

    def update_parameters(self, *args, **kwargs):
        """
        Update distribution parameters
        """
        raise NotImplementedError


class CgDistribution(Distribution):
    """
    Complex Gaussian Distribution (K classes, F bins)
    """
    def __init__(self, phi=None, covar=None):
        super(CgDistribution, self).__init__(covar)
        self.parameters["phi"] = phi

    def update_parameters(self, obs, gamma, force_hermitian=True):
        """
        Update phi & covar
        Args:
            obs: F x M x T
            gamma: K x F x T
        My modification: first update R then update phi
        so R is updated with the previous phi, updating ohi is for calculating gamma and R at next step
        """
        _, M, _ = obs.shape

        # update R
        denominator = np.sum(gamma, -1, keepdims=True)
        # K x F x M x M
        R = np.einsum("...t,...xt,...yt->...xy",
                      gamma / (self.parameters["phi"] + 1e-20), obs, obs.conj())
        R = R / (denominator[..., None] + 1e-20)
        self.update_covar(R, force_hermitian=force_hermitian)

        # update phi
        R_inv = self.covar(inv=True)
        phi = np.einsum("...xt,...xy,...yt->...t", obs.conj(), R_inv, obs)
        # phi = np.abs(phi)
        phi = np.maximum(np.abs(phi), EPSILON)  # -> log_pdf
        self.parameters["phi"] = phi / M

    def log_pdf(self, obs):
        """
        Formula:
            N(y, R) = e^(-y^H * R^{-1} * y) / det(pi*R)
        since:
            phi = trace(y * y^H * R^{-1}) / M = y^H * R^{-1} * y / M
        then:
            N(y, phi*R) = e^(-y^H * R^{-1} * y / phi) / det(pi*R*phi)
                        = e^{-M} / (det(R) * (phi * pi)^M)
        log N = const - log[det(R)] - M * log(phi)

        Arguments
            obs: mixture observation, F x M x T
        Return:
            logpdf: K x F x T
        """
        self.check_status()
        _, M, _ = obs.shape
        log_det = self.parameters["covar"].det(log=True)
        log_pdf = -M * np.log(self.parameters["phi"]) - log_det
        # K x F x T
        return log_pdf

    def pdf(self, obs):
        '''
        N = 1 / (const * det(R) * phi^M) * e^(-M)
        Args:
            obs: mixture observation, F x M x T
        Returns:
            pdf: K x F x T
        Note: only applicable to num_class=2
        '''
        self.check_status()
        _, M, _ = obs.shape
        det = self.parameters["covar"].det(log=False)   # [K, F, T]
        pdf = np.multiply(det, np.power(self.parameters["phi"], M))   # K x F x T
        pdf_sqrt = np.sqrt(pdf)

        return pdf_sqrt


class Cgmm(object):
    """
    Complex Gaussian Mixture Model (CGMM)
    """
    def __init__(self, mdl, alpha, Rn):
        self.cg = CgDistribution() if mdl is None else mdl
        # K x F
        self.alpha = alpha
        self.Rn = Rn

    def update(self, obs, gamma, update_alpha=False):
        """
        Update parameters in Cgmm
        Arguments:
            obs: mixture observation, F x M x T
            gamma: K x F x T
        """
        # update phi & R
        self.cg.update_parameters(obs, gamma)
        covar_tmp = self.cg.covar(inv=False)
        self.Rn = covar_tmp[1]
        # update alpha
        if update_alpha:
            self.alpha = np.mean(gamma, -1)

    def predict_log(self, obs, return_Q=False):
        """
        Compute gamma (posterior) using Cgmm
        Arguments:
            obs: mixture observation, F x M x T
        Return:
            gamma: posterior, K x F x T
        """
        # K x F x T
        log_pdf = self.cg.log_pdf(obs)

        log_pdf = log_pdf - np.amax(log_pdf, 0, keepdims=True)
        # K x F x T
        pdf = np.exp(log_pdf)
        # K x F x T
        nominator = np.sqrt(pdf * self.alpha[..., None])
        denominator = np.sum(nominator, 0, keepdims=True)
        gamma = nominator / np.maximum(denominator, EPSILON)

        Q = None
        if return_Q:
            # K x F x T => F x T
            pdf_ktf = np.log(np.exp(self.cg.log_pdf(obs)) * self.alpha[..., None])
            pdf_ktf_gamma = pdf_ktf * gamma
            # each TF-bin
            Q = np.mean(np.sum(pdf_ktf_gamma, 0))

        if return_Q:
            return gamma, Q
        else:
            return gamma

    def predict(self, obs, return_Q=False):
        """
        Compute gamma (posterior) using Cgmm
        Arguments:
        obs: mixture o
        bservation, F x M x T
        Return:
            gamma: posterior, K x F x T
        """
        # K x F x T
        pdf = self.cg.pdf(obs)

        # K x F x T
        nominator = pdf * self.alpha[..., None]
        denominator = np.sum(nominator, 0, keepdims=True)
        gamma = nominator / (denominator + 1e-20)

        # Note Since N = 1 / (const * det(R) * phi^M) * e^(-M), the pdf/gamma should be reversed
        tmp_gamma = copy.deepcopy(gamma[0])
        gamma[0, :, :] = gamma[1, :, :]
        gamma[1, :, :] = tmp_gamma

        Q = None
        if return_Q:
            # K x F x T => F x T
            pdf_ktf = np.log(pdf * self.alpha[..., None] + EPSILON)
            pdf_ktf_gamma = pdf_ktf * gamma
            # each TF-bin
            Q = np.mean(np.sum(pdf_ktf_gamma, 0))

        if return_Q:
            return gamma, Q
        else:
            return gamma

class CgmmTrainer(object):
    """
    Cgmm Trainer
    """
    def __init__(self,
                 obs,
                 num_classes,
                 logger,
                 gamma=None,
                 cgmm=None,
                 Rn=None,
                 update_alpha=False):
        """
        Arguments:
            obs: mixture observation, M x F x T
            gamma: initial gamma, K x F x T
        """
        self.update_alpha = update_alpha
        # F x M x T
        self.obs = np.einsum("mft->fmt", obs)
        F, M, T = self.obs.shape
        self.logger = logger
        self.logger.info(f"CGMM instance: F = {F:d}, T = {T:}, M = {M}")

        if cgmm is None:
            if num_classes == 2:
                if gamma is None:
                    Rs = np.einsum("...dt,...et->...de", self.obs,
                                   self.obs.conj()) / T
                    if Rn is None:
                        Rn = np.stack(
                            [np.eye(M, M, dtype=np.complex) for _ in range(F)])
                    R = np.stack([Rs, Rn])
                else:
                    gamma = np.stack([gamma, 1 - gamma])
            else:
                # random init gamma
                if gamma is None:
                    gamma = np.random.uniform(size=[num_classes, F, T])
                    gamma = gamma / np.sum(gamma, 0, keepdims=True)
                    logger.info(
                        f"Random initialized, num_classes = {num_classes}")
            if gamma is not None:
                den = np.maximum(np.sum(gamma, axis=-1, keepdims=True),
                                 EPSILON)
                # 2 x F x M x M
                R = np.einsum("...t,...xt,...yt->...xy", gamma, self.obs,
                              self.obs.conj()) / den[..., None]
            # init phi & R
            self.Rn = Rn
            R_inv = Covariance(R).mat(inv=True)
            phi = np.einsum("...xt,...xy,...yt->...t", self.obs.conj(), R_inv,
                            self.obs)  # -> [K, F, T]
            phi = np.maximum(np.abs(phi), EPSILON)   # -> log
            cg = CgDistribution(phi=phi / M, covar=R)
            alpha = np.ones([num_classes, F])   # / num_classes
            self.cgmm = Cgmm(cg, alpha, self.Rn)
            # self.gamma = self.cgmm.predict(self.obs)  # first E step

        else:
            with open(cgmm, "r") as pkl:
                self.cgmm = pickle.load(pkl)
            logger.info(f"Resume cgmm model from {cgmm}")
            # self.gamma = self.cgmm.predict(self.obs)  # first E step

    def train(self, num_iters):
        """
        Train in EM progress
        """
        for i in range(num_iters):
            # E step
            self.gamma, Q = self.cgmm.predict_log(self.obs, return_Q=True)
            self.logger.info(f"Iter {i + 1:2d}/{num_iters}: Q = {Q:.4f}")

            # M step
            self.cgmm.update(self.obs,
                             self.gamma,
                             update_alpha=self.update_alpha)
            self.Rn = self.cgmm.Rn

        return self.gamma, self.Rn