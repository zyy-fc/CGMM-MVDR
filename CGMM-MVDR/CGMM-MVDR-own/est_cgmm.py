"""
Estimate the complex GMM parameters and generate the mask for noise only and noisy t-f bins.
"""
import numpy as np
import math
import scipy.io as sio

d = 1 / math.pow(math.sqrt(math.pi * 2), 5)
beta = 1/math.pow(10, 6)

class CGMM(object):
    def __init__(self, num_bins, time_steps, num_channels):
        """
            num_bins:   number of bins along frequent axis(usually 257)
            time_steps: number of frames per channel
            num_channels: number of channels, equals GMM dim
        """
        self.num_bins, self.time_steps = num_bins, time_steps
        self.dim = num_channels
        # lambda, pi, R for noisy/noise part
        self.lambda_ = np.zeros([time_steps, num_bins]).astype(np.complex)
        self.phi = np.ones([time_steps, num_bins]).astype(np.complex)
        self.sigma = np.ones([num_channels, num_channels, num_bins]).astype(np.complex)
        self.af = 1
        self.posterior = np.zeros([self.num_bins, self.time_steps]).astype(np.complex)

    def covar_entropy(self):
        """
            Return entropy among eigenvalues of correlation matrix on
            each frequency bin.
        """
        entropy = []
        for sigma in self.sigma:
            egval, _ = np.linalg.eig(sigma)
            real_eigen = egval.real / egval.real.sum()
            entropy.append(-(real_eigen * np.log(real_eigen)).sum())
        return entropy

    # def init_sigma(self, sigma):
    #     """
    #     Inputs : sigma is a np.matrix list
    #     Keeps \sigma^{-1} and det(\sigma), \sigma equals \mean(y^H * y)
    #     """
    #     assert type(sigma) == list
    #     self.sigma_inv = [mat.I for mat in sigma]
    #     self.sigma_det = [np.linalg.det(mat) for mat in sigma]


class CGMMTrainer(object):
    def __init__(self, num_bins, time_steps, num_channels):
        self.noise_part = CGMM(num_bins, time_steps, num_channels)
        self.noisy_part = CGMM(num_bins, time_steps, num_channels)
        self.covar = np.zeros([num_channels, num_channels, time_steps, num_bins]).astype(np.complex)
        self.num_bins = num_bins
        self.time_steps = time_steps

    def init_sigma(self, spectrum):
        """
            covar: precomputed correlation matrix of each channel
            Here we init noisy_part'R as correlation matrix of observed signal and noise_part'R as unit matrx
        """
        print("initialize sigma ..")
        time_steps, num_bins, num_channels = spectrum.shape
        for f in range(num_bins):
            for t in range(time_steps):
                y = np.matrix(spectrum[t, f].reshape(num_channels, 1))
                self.covar[:, :, t, f] = y * y.H
                self.noisy_part.sigma[:, :, f] += self.covar[:, :, t, f]
            self.noisy_part.sigma[:, :, f] /= time_steps
            self.noise_part.sigma[:, :, f] = np.matrix(np.eye(num_channels, num_channels)).astype(np.complex)

    def train(self, spectrums, iters=10):
        print("start training ..")
        self.init_sigma(spectrums)
        time_steps, num_bins, num_channels = spectrums.shape
        p_noise = np.ones([time_steps, num_bins]).astype(complex)
        p_noisy = np.ones([time_steps, num_bins]).astype(complex)

        for it in range(1, iters+1):
            for f in range(num_bins):
                R_noisy_onbin = np.matrix(self.noisy_part.sigma[:, :, f])
                R_noise_onbin = np.matrix(self.noise_part.sigma[:, :, f])

                # R_noisy_onbin += beta * np.matrix(np.eye(num_channels, num_channels))
                # R_noise_onbin += beta * np.matrix(np.eye(num_channels, num_channels))

                R_noisy_inv = R_noisy_onbin.I
                R_noise_inv = R_noise_onbin.I

                R_noisy_accu = np.zeros([num_channels, num_channels]).astype(np.complex)
                R_noise_accu = np.zeros([num_channels, num_channels]).astype(np.complex)

                for t in range(time_steps):
                    corre = self.covar[:, :, t, f]
                    obs = np.matrix(spectrums[t, f].reshape(num_channels, 1))

                    # update phi
                    self.noisy_part.phi[t, f] = np.trace(corre * R_noisy_inv) / num_channels
                    self.noise_part.phi[t, f] = np.trace(corre * R_noise_inv) / num_channels

                    # update lambda
                    k_noise = obs.H * (R_noise_inv / self.noise_part.phi[t, f]) * obs
                    k_noise = complex(k_noise)
                    det_noise = np.linalg.det(self.noise_part.phi[t, f] * R_noise_onbin)
                    p_noise[t, f] = np.exp((-1)*k_noise) / (math.pi*det_noise)

                    k_noisy = obs.H * (R_noisy_inv / self.noisy_part.phi[t, f]) * obs
                    k_noisy = complex(k_noisy)
                    det_noisy = np.linalg.det(self.noisy_part.phi[t, f] * R_noisy_onbin)
                    p_noisy[t, f] = np.exp((-1) * k_noisy) / (math.pi*det_noisy)

                    self.noise_part.lambda_[t, f] = self.noise_part.af * p_noise[t, f] / \
                                            (self.noise_part.af * p_noise[t, f] + self.noisy_part.af * p_noisy[t, f])
                    self.noisy_part.lambda_[t, f] = self.noisy_part.af * p_noisy[t, f] / \
                                            (self.noise_part.af * p_noise[t, f] + self.noisy_part.af * p_noisy[t, f])

                    R_noise_accu += (self.noise_part.lambda_[t, f] / self.noise_part.phi[t, f]) * corre
                    R_noisy_accu += (self.noisy_part.lambda_[t, f] / self.noisy_part.phi[t, f]) * corre

                tmp_noise_sigma = R_noise_accu / sum(self.noise_part.lambda_[:, f])
                tmp_noisy_sigma = R_noisy_accu / sum(self.noisy_part.lambda_[:, f])

                self.noise_part.af = sum(self.noise_part.lambda_[:, f]) / time_steps
                self.noisy_part.af = sum(self.noisy_part.lambda_[:, f]) / time_steps

                sio.savemat('tmp_noise_sigma.mat', {'tmp_noise_sigma': tmp_noise_sigma})
                egval1, _ = np.linalg.eig(tmp_noise_sigma)
                real_eigen1 = egval1.real / egval1.real.sum()
                en_n = -(real_eigen1 * np.log(real_eigen1)).sum()

                egval2, _ = np.linalg.eig(tmp_noisy_sigma)
                real_eigen2 = egval2.real / egval2.real.sum()
                en_y = -(real_eigen2 * np.log(real_eigen2)).sum()

                if en_y > en_n:
                    self.noise_part.sigma[:, :, f] = tmp_noisy_sigma
                    self.noisy_part.sigma[:, :, f] = tmp_noise_sigma
                else:
                    self.noise_part.sigma[:, :, f] = tmp_noise_sigma
                    self.noisy_part.sigma[:, :, f] = tmp_noisy_sigma

            Q = np.sum(self.noise_part.lambda_ * np.log(p_noise + 0.001) + \
                        self.noisy_part.lambda_ * np.log(p_noisy + 0.001)) / (time_steps*num_bins)

            print('epoch {0:2d}: Likelihood = ({1.real:.5f}, {1.imag:.5f}i)'.format(it, Q))














