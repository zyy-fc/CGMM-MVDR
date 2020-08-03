#!/usr/bin/env python

# zyy@2020

import argparse
import numpy as np

from pathlib import Path
from libs.cluster import CgmmTrainer, permu_aligner
from libs.data_handler import SpectrogramReader, ScriptReader, NumpyReader, NumpyWriter
from libs.utils import get_logger
from libs.opts import StftParser, StrToBoolAction
from libs.conf import stft_conf, cgmm_conf

logger = get_logger('./log/run_cgmm.log', file=True)

def run(stft_kwargs, wav_scp, dst_dir, num_iters, num_classes, seed, init_mask, solve_permu, update_alpha, fmt):
    np.random.seed(seed)
    spectrogram_reader = SpectrogramReader(wav_scp, **stft_kwargs)
    MaskReader = {"numpy": NumpyReader, "kaldi": ScriptReader}
    init_mask_reader = MaskReader[fmt](init_mask) if init_mask else None

    num_done = 0
    with NumpyWriter(dst_dir) as writer:
        dst_dir = Path(dst_dir)
        for key, stft in spectrogram_reader:
            if not (dst_dir / f"{key+'.speech'}.npy").exists():
            # if not dst_dir.exists():
                init_mask = None
                if init_mask_reader and key in init_mask_reader:
                    init_mask = init_mask_reader[key]
                    # T x F => F x T
                    if init_mask.ndim == 2:
                        init_mask = np.transpose(init_mask)
                    else:
                        init_mask = np.transpose(init_mask, (0, 2, 1))
                    logger.info("Using external TF-mask to initialize cgmm")
                # stft: N x F x T
                trainer = CgmmTrainer(stft,
                                      num_classes,
                                      logger,
                                      gamma=init_mask,
                                      update_alpha=update_alpha)
                try:
                    masks = trainer.train(num_iters)
                    # K x F x T => K x T x F
                    masks = np.transpose(masks, (0, 2, 1))
                    num_done += 1
                    if solve_permu:
                        masks = permu_aligner(masks)
                        logger.info(
                            "Permutation alignment done on each frequency")
                        writer.write(key, masks.astype(np.float32))
                    if num_classes == 2:
                        # masks = masks[0]
                        writer.write(key+'.speech', masks[0].astype(np.float32))
                        writer.write(key+'.noise', masks[1].astype(np.float32))
                    # writer.write(key, masks.astype(np.float32))
                    logger.info(f"Training utterance {key} ... Done")
                except RuntimeError:
                    logger.warn(f"Training utterance {key} ... Failed")
            else:
                logger.info(f"Training utterance {key} ... Skip")
    logger.info(
        f"Train {num_done:d} utterances over {len(spectrogram_reader):d}")


if __name__ == "__main__":
    run(stft_conf, **cgmm_conf)