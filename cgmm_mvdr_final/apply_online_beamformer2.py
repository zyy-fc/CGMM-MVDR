#!/usr/bin/env python
# coding=utf-8
# zyy@2020
#
"""
Do mvdr/... adaptive beamformer
"""

import argparse

import numpy as np
from scipy.io import loadmat

from pathlib import Path
from libs.utils import inverse_stft_reserve, get_logger, nextpow2, cmat_abs
from libs.cluster import CgmmTrainer, permu_aligner
from libs.data_handler import WaveWriter, SegmentSpecReader, WaveReader
from libs.beamformer import OnlineGevdBeamformer, OnlineMvdrBeamformer
from libs.conf import beamformer_online_conf, stft_conf
from visualize_tf_matrix import save_figure
import scipy.io as scio

logger = get_logger('./log/beamformer_mvdr_online2.log', file=True)
beamformers = ["mvdr", "mpdr", "mpdr-whiten", "gevd", "pmwf-0", "pmwf-1"]

def filter_masks(stft_mat, speech_mask, interf_mask, vad_proportion):
    '''
    Make sure the speech_masks/inter_masks in T X F
    Returns:
        speech_masks, interf_masks -> T X F
    '''
    _, F, _ = stft_mat.shape
    # if in F x T, convert it into T X F
    if interf_mask.shape[0] == F and interf_mask.shape[1] != F:
        interf_mask = np.transpose(interf_mask)
        if speech_mask is not None:
            speech_mask = np.transpose(speech_mask)
    if 0.5 < vad_proportion < 1:
        vad_mask, N = compute_vad_masks(stft_mat[0], vad_proportion)
        logger.info(f"Filtering {N} TF-masks...")
        interf_mask = np.where(vad_mask, 1.0e-4, interf_mask)
        if speech_mask is not None:
            speech_mask = np.where(vad_mask, 1.0e-4, speech_mask)
    if speech_mask is None:
        speech_mask = 1-interf_mask
    return speech_mask, interf_mask


def compute_vad_masks(spectrogram, proportion):
    """
    We ignore several minimum values and keep proportion*100% energy
    Arguments:
        spectrogram: F x T
    Return:
        vad_mask: T x F
    """
    energy_mat = cmat_abs(spectrogram)
    energy_vec = np.sort(energy_mat.flatten())
    filter_energy = np.sum(energy_vec) * (1 - proportion)
    threshold = 0
    cumsum, index = 0, 0
    while index < energy_vec.shape[0]:
        threshold = energy_vec[index]
        cumsum += threshold
        if cumsum > filter_energy:
            break
        index += 1
    # silence if 1
    vad_mask = (energy_mat < threshold)
    return vad_mask.transpose(), index


def estimate_masks(stft_mat, init_mask, Rn_init, update_alpha, num_iters, key, num_classes=2):
    '''
    Estimate masks using cgmm method with constraint num_classes=2
    The meaning of params can be referred to estimate_cgmm_masks.py
    Args:
        stft_mat: N x F x T

    Returns:
        masks -> [masks[0](speech), masks[1](noise)]
    '''
    trainer = CgmmTrainer(stft_mat, num_classes,
                          logger, gamma=init_mask, Rn=Rn_init, update_alpha=update_alpha)
    try:
        masks, Rn = trainer.train(num_iters)
        # K x F x T => K x T x F
        masks = np.transpose(masks, (0, 2, 1))
        logger.info(f"Training utterance {key} ... Done")
        return masks, Rn
    except RuntimeError:
        logger.warn(f"Training utterance {key} ... Failed")

def run(stft_kwargs, block_size, wav_scp, dst_dir, beamformer, sr, ban, vad_proportion, alpha,
        chunk_size, channels, init_mask, mask_alpha, num_iters, solve_permu):
    round_power_of_two = stft_kwargs.pop('round_power_of_two')
    frame_len = stft_kwargs['frame_len']
    frame_hop = stft_kwargs['frame_hop']

    num_bins = nextpow2(frame_len) // 2 + 1

    if chunk_size < 32:
        raise RuntimeError(f"Seems chunk size({chunk_size:.2f}) " +
                           "too small for online beamformer")

    beamformer = OnlineMvdrBeamformer(num_bins, channels, alpha)
    logger.info(f"Using online {beamformer} beamformer, " +
                f"chunk size = {chunk_size:d}")

    wave_reader = WaveReader(wav_scp, normalize=True)
    SegSpectrogram_reader = SegmentSpecReader(
        wav_scp, normalize=True,
        round_power_of_two=round_power_of_two, **stft_kwargs)

    num_done = 0
    with WaveWriter(dst_dir, sr=sr) as writer:
        for key, samps in wave_reader:
            logger.info(f"Processing utterance {key}")
            xlen = samps.shape[-1]
            nframes = int((xlen - frame_len)/frame_hop) + 1

            # init params
            start_idx, cnt = 0, 0
            enh_samps, interf_mask, speech_mask  = [], [], []
            Rn_init = None
            reserve_init = None

            stft_mat = np.zeros((channels, num_bins, chunk_size), dtype=np.complex64)
            count = 0
            while cnt < nframes :
                print('count is', count)
                end_idx = start_idx + block_size
                # read data and stft
                stft_mat_c = SegSpectrogram_reader._load(key, beg=start_idx, end=end_idx)  # N x F x T
                nframe_c = stft_mat_c.shape[-1]

                # update the stft_mat
                stft_mat[:, :, :-nframe_c] = stft_mat[:, :, nframe_c:]
                stft_mat[:, :, -nframe_c:] = stft_mat_c

                # estimate masks using cgmm online -> K x T x F
                try:
                    masks_c, Rn_c = estimate_masks(stft_mat, init_mask, Rn_init, mask_alpha,
                                           num_iters, key, num_classes=2)
                    Rn_init = Rn_c
                except RuntimeError:
                    logger.warn(f"Training utterance {key} in {cnt}-th block ... Failed")
                    continue

                if solve_permu:
                    masks = permu_aligner(masks)
                    logger.info("Permutation alignment done on each frequency")

                # only store the interf_mask
                speech_mask_c, interf_mask_c = filter_masks(stft_mat, None, masks_c[1], vad_proportion)
                interf_mask.append(interf_mask_c[-nframe_c:, :])   # T X F
                speech_mask.append(speech_mask_c[-nframe_c:, :])

                # online mvdr beamforming
                chunk = beamformer.run(speech_mask_c, stft_mat, mask_n=interf_mask_c, ban=ban) # [F, chunk_size]
                enh_chunks, reserve_c = inverse_stft_reserve(chunk[:, -nframe_c:], reserve=reserve_init, **stft_kwargs)
                enh_samps.append(enh_chunks)
                reserve_init = reserve_c
                # enh_samps.append(chunk[:, -nframe_c:])   # -> add the frame_chunk

                start_idx = end_idx
                cnt += nframe_c
                count += 1

            # enh_samps = np.hstack(enh_samps)
            # enh_samps = inverse_stft(enh_samps, **stft_kwargs)   # -> convert the entire signal
            enh_samps = np.hstack(enh_samps)
            # my modification to modify the amplitude
            enh_samps = enh_samps   # orig is : samps = samps * norm

            save_figure(key, np.real(np.vstack(speech_mask)), Path(dst_dir) / (key + '.log_frame_ite').replace(".", "-"),
                        cmap='binary', title='speech_mask')
            writer.write(key+'_log_frame_ite', enh_samps)

            num_done += 1
    logger.info(f"Processed {num_done:d} utterances " +
                f"out of {len(wave_reader):d}")

if __name__ == "__main__":
    block_size = 8000
    run(stft_conf, block_size, **beamformer_online_conf)
