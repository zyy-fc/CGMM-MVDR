
# cgmm configure
cgmm_conf = {'wav_scp': './data/input/mix.scp',  # Multi-channel wave scripts in kaldi format -> str
             'dst_dir': './data/cgmm_out',  # Location to dump estimated speech masks -> str
             'num_iters': 50,
             'num_classes': 2,
             'seed': 777,  # Random seed for initialization
             'init_mask': "", # Initial TF-mask for cgmm initialization
             'solve_permu': False,  # If true, solving permutation problems
             'update_alpha': True,
             'fmt': 'numpy',
             }


# stft configure
stft_conf = {'frame_len': 1024,
             'frame_hop':256,
             "round_power_of_two": True,
             "window": 'hann',
             "center":False,
             "transpose": False
             }

# beamformer config
beamformer_conf = {'wav_scp': './data/input/mix.scp',
                   'tgt_mask': './data/cgmm_out/cgmm_speech_masks.scp', # Scripts of target masks in kaldi's archive or numpy's ndarray
                   'dst_dir': './data/beamformer_out/mvdr/online',
                   'itf_mask': './data/cgmm_out/cgmm_noise_masks.scp',
                   'fmt': 'numpy',
                   'beamformer': 'mvdr',
                   'pmwf_ref': -1,
                   'sr': 16000,   # sample rate
                   'ban': False,
                   'rank1_appro': 'none',
                   'mask': False,   # post-masking : Masking enhanced spectrogram after beamforming or not
                   'vad_proportion': 1,  # Energy proportion to filter silence masks [0.5, 1]
                   'alpha': 0.8,  # Online Remember coefficient when updating covariance matrix
                   'chunk_size': 50,
                   'channels': 8  # Number of channels available
                   }

# frame-beamformer config
beamformer_frame_conf = {'wav_scp': './data/input/mix.scp',
                   'tgt_mask': './data/cgmm_out/cgmm_speech_masks.scp', # Scripts of target masks in kaldi's archive or numpy's ndarray
                   'dst_dir': './data/beamformer_out/mvdr/frame',
                   'fmt': 'numpy',
                   'sr': 16000,   # sample rate
                   'mask': False,   # post-masking : Masking enhanced spectrogram after beamforming or not
                   'vad_proportion': 1,  # Energy proportion to filter silence masks [0.5, 1]
                   'channels': 6  # Number of channels available
                   }


# beamformer and mask online config
beamformer_online_conf = {'wav_scp': './data/input/mix.scp',
                   'dst_dir': './data/beamformer_out/mvdr/online_mask_mvdr2',
                   'beamformer': 'mvdr',
                   'sr': 16000,   # sample rate
                   'ban': False,
                   'vad_proportion': 1,  # Energy proportion to filter silence masks [0.5, 1]
                   'alpha': 0.8,  # Online Remember coefficient when updating covariance matrix
                   'chunk_size': 50,
                   'channels': 8,  # Number of channels available
                   'init_mask': None, # Initial TF-mask for cgmm initialization
                   'mask_alpha': False,
                   'num_iters': 10,
                   'solve_permu': False  # If true, solving permutation problems
                   }

# visualize kaldi's features/numpy's ndarray on T-F domain
visualize_masks_conf = {'rspec': './data/cgmm_out',  # Read specifier of archives or directory of ndarrays
                        'input_type': 'dir',  # Type of the input read specifier
                        'frame_hop': 256,
                        'sr': 16000,
                        'cache_dir': './data/visualization',        # "Directory to cache pictures
                        'apply_log': False,    # Apply log on input features
                        'trans': False,    # Apply matrix transpose on input features
                        'norm': False,  # Normalize values in [-1, 1] before visualization
                        'cmap': 'jet',  # Colormap used when save figures
                        'index': -1,    # Channel index to plot, -1 means all
                        'title': 'cgmm_mask',  # Title of the pictures
                        'split': -1,  # Split 2D matrice into several sub-matrices
                        }










