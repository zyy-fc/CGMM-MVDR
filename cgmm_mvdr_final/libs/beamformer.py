#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import numpy as np
import scipy as sp

from .utils import EPSILON, cmat_abs
from scipy.linalg import toeplitz
"""
Implement for some classic beamformer
"""


def do_ban(weight, Rn):
    """
    Do Blind Analytical Normalization(BAN)
    Arguments: (for N: num_mics, F: num_bins)
        weight: shape as F x N
        Rn: shape as F x N x N
    Return:
        ban_weight: shape as F x N
    """
    nominator = np.einsum("...a,...ab,...bc,...c->...", np.conj(weight), Rn,
                          Rn, weight)
    denominator = np.einsum("...a,...ab,...b->...", np.conj(weight), Rn,
                            weight)
    filters = np.sqrt(cmat_abs(nominator)) / np.maximum(
        np.real(denominator), EPSILON)
    return filters[:, None] * weight


def solve_pevd(Rs, Rn=None):
    """
    Return principle eigenvector of covariance matrix (pair)
    Arguments: (for N: num_mics, F: num_bins)
        Rs: shape as F x N x N
        Rn: same as Rs if not None
    Return:
        pvector: shape as F x N
    """
    if Rn is None:
        # batch(faster) version
        # eigenvals: F x N, ascending order
        # eigenvecs: F x N x N on each columns, |vec|_2 = 1
        # NOTE: eigenvalues computed by np.linalg.eig is not necessarily ordered.
        _, eigenvecs = np.linalg.eigh(Rs)
        return eigenvecs[:, :, -1]
    else:
        F, N, _ = Rs.shape
        pvec = np.zeros((F, N), dtype=np.complex)
        for f in range(F):
            try:
                # sp.linalg.eigh returns eigen values in ascending order
                _, eigenvecs = sp.linalg.eigh(Rs[f], Rn[f])
                pvec[f] = eigenvecs[:, -1]
            except np.linalg.LinAlgError:
                try:
                    eigenvals, eigenvecs = sp.linalg.eig(Rs[f], Rn[f])
                    pvec[f] = eigenvecs[:, np.argmax(eigenvals)]
                except np.linalg.LinAlgError:
                    raise RuntimeError(
                        "LinAlgError when computing eig on frequency "
                        "{f}: \nRs = {Rs[f]}, \nRn = {Rn[f]}")
        return pvec


def compute_covar(obs, tf_mask):
    """
    Arguments: (for N: num_mics, F: num_bins, T: num_frames)
        tf_mask: shape as T x F, same shape as network output
        obs: shape as N x F x T
    Return:
        covar_mat: shape as F x N x N
    """
    # num_bins x num_mics x num_frames
    obs = np.transpose(obs, (1, 0, 2))
    # num_bins x 1 x num_frames
    mask = np.expand_dims(np.transpose(tf_mask), axis=1)
    denominator = np.maximum(np.sum(mask, axis=-1, keepdims=True), 1e-6)
    # num_bins x num_mics x num_mics
    covar_mat = np.einsum("...dt,...et->...de", mask * obs,
                          obs.conj()) / denominator
    return covar_mat

def smmoth_covar(covar_mat, len_win):
    F, N, _ = covar_mat.shape
    covar_mat = np.reshape(covar_mat, (F, N*N))
    covar_mat = np.dot(get_win(len_win, F), covar_mat)
    return np.reshape(covar_mat, (F, N, N))

def get_win(len_win, F):
    half_win = 0.5 * np.cos((np.arange(len_win+1) / (len_win+1) * np.pi)) + 0.5
    win = toeplitz(np.concatenate((half_win, np.zeros(F-len_win-1)), axis=0))
    win_sum = np.reshape(np.sum(np.abs(win), axis=1), (win.shape[0]))
    win_sum = np.tile(win_sum, (win.shape[1], 1))
    win = win / np.transpose(win_sum)
    return win


def beam_pattern(weight, steer_vector):
    """
    Compute beam pattern of the fixed beamformer
    Arguments (for N: num_mics, F: num_bins, D: num_doas, B: num_beams)
        weight: B x F x N or F x N (single or multiple beams)
        steer_vector: F x D x N
    Return
        pattern: [F x D, ...] or F x D
    """

    if weight.shape[-1] != steer_vector.shape[-1] or weight.shape[
            -2] != steer_vector.shape[0]:
        raise RuntimeError("Shape mismatch between weight and steer_vector")

    def single_beam(weight, sv):
        # F x D x 1
        bp = sv @ np.expand_dims(weight.conj(), -1)
        return np.squeeze(np.abs(bp))

    if weight.ndim == 2:
        return single_beam(weight, steer_vector)
    elif weight.ndim == 3:
        return [single_beam(w, steer_vector) for w in weight]
    else:
        raise RuntimeError(f"Expect 2/3D beam weights, got {weight.ndim}")



class Beamformer(object):
    def __init__(self):
        pass

    def beamform(self, weight, obs):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            weight: shape as F x N
            obs: shape as N x F x T
        Return:
            stft_enhan: shape as F x T
        """
        # N x F x T => F x N x T
        if weight.shape[0] != obs.shape[1] or weight.shape[1] != obs.shape[0]:
            raise ValueError("Input obs do not match with weight, " +
                             f"{weight.shape} vs {obs.shape}")
        obs = np.transpose(obs, (1, 0, 2))
        obs = np.einsum("...n,...nt->...t", weight.conj(), obs)
        return obs


class SupervisedBeamformer(Beamformer):
    """
    BaseClass for TF-mask based beamformer
    """
    def __init__(self, num_bins):
        super(SupervisedBeamformer, self).__init__()
        self.num_bins = num_bins

    def compute_covar_mat(self, target_mask, obs):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            target_mask: shape as T x F, same shape as network output
            obs: shape as N x F x T
        Return:
            covar_mat: shape as F x N x N
        """
        if target_mask.shape[1] != self.num_bins or target_mask.ndim != 2:
            raise ValueError(
                "Input mask matrix should be shape as " +
                f"[num_frames x num_bins], now is {target_mask.shape}")
        if obs.shape[1] != target_mask.shape[1] or obs.shape[
                2] != target_mask.shape[0]:
            raise ValueError(
                "Shape of input obs do not match with " +
                f"mask matrix, {obs.shape} vs {target_mask.shape}")
        return compute_covar(obs, target_mask)

    def weight(self, Rs, Rn):
        """
        Need reimplement for different beamformer
        """
        raise NotImplementedError

    def run(self, mask_s, obs, mask_n=None, ban=False):
        """
        Run beamformer based on TF-mask
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            mask_s: shape as T x F, same shape as network output
            obs: shape as N x T x F
        Returns:
            stft_enhan: shape as F x T
        """
        Rn = self.compute_covar_mat(1 - mask_s if mask_n is None else mask_n,
                                    obs)
        Rs = self.compute_covar_mat(mask_s, obs)
        weight = self.weight(Rs, Rn)
        return self.beamform(do_ban(weight, Rn) if ban else weight, obs)


class OnlineSupervisedBeamformer(SupervisedBeamformer):
    """
    Online version of SupervisedBeamformer
    """
    def __init__(self, num_bins, num_channels, alpha=0.8):
        super(OnlineSupervisedBeamformer, self).__init__(num_bins)
        self.covar_mat_shape = (num_bins, num_channels, num_channels)
        self.reset_stats(alpha=alpha)
        nfft = (num_bins-1)*2
        self.win_s = int((2 * nfft) / 512)   # self.win_n = self.win_s

    def reset_stats(self, alpha=0.8):
        # self.Rs = np.einsum("...dt,...et->...de", obs, obs.conj())
        self.Rs = np.zeros(self.covar_mat_shape, dtype=np.complex)
        # [np.matrix(np.eye(num_channels, num_channels).astype(np.complex)) for f in range(num_bins)]
        self.Rn = np.zeros(self.covar_mat_shape, dtype=np.complex)

        self.alpha = alpha
        self.reset = True

    def run(self, mask_s, obs, mask_n=None, ban=False):
        """
        Run beamformer based on TF-mask, online version
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            mask_s: shape as T x F, same shape as network output
            obs: shape as N x T x F
        Returns:
            stft_enhan: shape as F x T
        """
        Rn = self.compute_covar_mat(1 - mask_s if mask_n is None else mask_n,
                                    obs)
        Rn = smmoth_covar(Rn, self.win_s)
        Rs = self.compute_covar_mat(mask_s, obs)
        Rs = smmoth_covar(Rs, self.win_s)

        # update stats
        phi = 1 if self.reset else (1 - self.alpha)
        self.Rs = self.Rs * self.alpha + phi * Rs
        self.Rn = self.Rn * self.alpha + phi * Rn
        # do beamforming
        weight = self.weight(self.Rs, self.Rn)
        return self.beamform(do_ban(weight, Rn) if ban else weight, obs)

class MvdrBeamformer(SupervisedBeamformer):
    """
    MVDR (Minimum Variance Distortionless Response) Beamformer
    Formula:
        h_mvdr(f) = R(f)_{vv}^{-1}*d(f) / [d(f)^H*R(f)_{vv}^{-1}*d(f)]
    where
        d(f) = P(R(f)_{xx}) P: principle eigenvector
    """
    def __init__(self, num_bins):
        super(MvdrBeamformer, self).__init__(num_bins)

    def weight(self, Rs, Rn):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rs: shape as F x N x N
            Rn: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        steer_vector = solve_pevd(Rs)
        # Rn*numerator = steer_vector -> numerator = Rn^(-1)*steer_vector
        numerator = np.linalg.solve(Rn, steer_vector)
        denominator = np.einsum("...d,...d->...", steer_vector.conj(),
                                numerator)
        return numerator / np.expand_dims(denominator, axis=-1)


class OnlineMvdrBeamformer(OnlineSupervisedBeamformer):
    """
    Online version of MVDR beamformer
    """
    def __init__(self, num_bins, num_channels, alpha=0.8):
        super(OnlineMvdrBeamformer, self).__init__(num_bins,
                                                   num_channels,
                                                   alpha=alpha)

    def weight(self, Rs, Rn):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rs: shape as F x N x N
            Rn: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        steer_vector = solve_pevd(Rs)
        numerator = np.linalg.solve(Rn, steer_vector)
        denominator = np.einsum("...d,...d->...", steer_vector.conj(),
                                numerator)
        return numerator / np.expand_dims(denominator, axis=-1)