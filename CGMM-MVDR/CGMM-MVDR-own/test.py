# ZYY@19/4/24
import numpy as np
import utils
import est_cgmm
from est_cgmm import CGMMTrainer
from utils import WaveWrapper
import beamformer
import scipy.io as sio
import wave
from utils import MultiChannelWrapper
import matplotlib.pyplot as plt

# Load the test multi-channel test data and do stft
wrapper = MultiChannelWrapper('./audio/audio_noisy.txt')
(time_steps, num_bins), spectrums = wrapper.spectrums(transpose=False)
num_channels = len(spectrums)
spectrums = np.array(spectrums).transpose(1, 2, 0)
# print(spectrums.shape)


# Estimate the TF-SPP and spacial covariance matrix for noisy speech and noise
trainer = CGMMTrainer(num_bins, time_steps, num_channels)
trainer.train(spectrums, iters=10)
sio.savemat('specs_enhan.mat', {'lambda_noise': trainer.noise_part.lambda_,
                                'lambda_noisy': trainer.noisy_part.lambda_,
                                'sigma_noise': trainer.noise_part.sigma,
                                'sigma_noisy': trainer.noisy_part.sigma})

# plot the noisy spectrum and the noise/noisy mask
wrapper_noisy_single = WaveWrapper('./audio/F01_22HC010C_STR.CH1.wav')
spectrums_noisy_single = utils.compute_spectrum(wrapper_noisy_single)

# wrapper_clean_single = WaveWrapper('./audio/F01_22HC010C_STR.CH1.Clean.wav')
# spectrums_clean_single = utils.compute_spectrum(wrapper_clean_single)
# utils.plot_spectrum(spectrums_noisy_single, wrapper_noisy_single.frame_duration, 'noisy.wav')
# utils.plot_spectrum(trainer.noisy_part.lambda_, wrapper_noisy_single.frame_duration, 'noisy.mask')
# utils.plot_spectrum(trainer.noise_part.lambda_, wrapper_noisy_single.frame_duration, 'noise.mask')

# apply the beamformer
save_dir = './audio/enh_mask.wav'
specs_enhan = np.zeros([time_steps, num_bins]).astype(np.complex)
clean_sigma = trainer.noisy_part.sigma - trainer.noise_part.sigma
for f in range(num_bins):
    steer_vector = beamformer.main_egvec(clean_sigma[:, :, f])
    specs_enhan[:, f] = beamformer.apply_mvdr(steer_vector, trainer.noise_part.sigma[:, :, f],
                                             spectrums[:, f, :])

# sio.savemat('specs_enhan.mat', {'specs': specs_enhan})
utils.plot_spectrum(specs_enhan, wrapper_noisy_single.frame_duration, 'enh_mask.wav')
utils.reconstruct_wave(specs_enhan, save_dir, filter_coeff=1)

# # apply the beamformer based on the ideal noise/speech sigma
# wrapper_clean = MultiChannelWrapper('./audio/audio_speech.txt')
# (time_steps, num_bins), spectrums_clean = wrapper_clean.spectrums(transpose=False)
# spectrums_clean = np.array(spectrums_clean).transpose(1, 2, 0)
# sigma_clean = np.ones([num_channels, num_channels, num_bins]).astype(np.complex)
#
# wrapper_noise = MultiChannelWrapper('./audio/audio_noise.txt')
# (time_steps, num_bins), spectrums_noise = wrapper_noise.spectrums(transpose=False)
# spectrums_noise = np.array(spectrums_noise).transpose(1, 2, 0)
# sigma_noise = np.ones([num_channels, num_channels, num_bins]).astype(np.complex)
#
# save_dir = './audio/enh_ideal.wav'
# for f in range(num_bins):
#     for t in range(time_steps):
#         y_clean = np.matrix(spectrums_clean[t,f])
#         sigma_clean[:, :, f] += y_clean.H * y_clean
#
#         y_noise = np.matrix(spectrums_noise[t, f])
#         sigma_noise[:, :, f] += y_noise.H * y_noise
#
#     sigma_clean[:, :, f] /= time_steps
#     sigma_noise[:, :, f] /= time_steps
#
# specs_enhan_ideal = np.zeros([time_steps, num_bins]).astype(np.complex)
# for f in range(num_bins):
#     steer_vector = beamformer.main_egvec(sigma_clean[:,:,f])
#     specs_enhan_ideal[:,f] = beamformer.apply_mvdr(steer_vector, sigma_noise[:, :, f],
#                                              spectrums[:, f, :])
#
# utils.plot_spectrum(specs_enhan_ideal, wrapper_noisy_single.frame_duration, 'enh_ideal.wav')
# utils.plot_spectrum(spectrums_clean_single, wrapper_clean_single.frame_duration, 'clean.wav')
# utils.plot_spectrum(spectrums_noisy_single, wrapper_noisy_single.frame_duration, 'noisy.wav')
# utils.reconstruct_wave(specs_enhan_ideal, save_dir, filter_coeff=1)
#



