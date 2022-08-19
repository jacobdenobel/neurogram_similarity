import numpy as np
import os
import glob
from scipy import signal
import matplotlib.pyplot as plt

# based on https://www.sciencedirect.com/science/article/pii/S0165027021001473

DATA_DIR = os.path.join(
    os.path.realpath(os.path.dirname(os.path.dirname(__file__))), "data"
)
DROPBOX_DIR = os.path.join(DATA_DIR, "dropbox")

def load_spiking(str):
    data_dir = os.path.join(DROPBOX_DIR, "spike_matrices")
    str_loc = glob.glob(os.path.join(data_dir, '*spike_matrix_' + str + '*'))
    pattern = np.load(str_loc[-1]) # pick last one
    return pattern


##############################
# PREPROCESSING
# FT = 10 us per bin
# MR = 100 us per bin

def preprocessing(pattern, scaling = 1): 
    visualize_MR = True
    visualize_FT = True
    fibres, num_samples = pattern.shape
    
    # normalize
    pattern -= np.mean(pattern)
    pattern /= np.std(pattern)

    length_MR = 128 # Hamming window mean-rate / envelope
    length_FT = 32 # Hamming window fine-timing / TFS
    
    # rebinning from 10 us to 100
    reduction_in_size = 10
    new_num_samples = int(np.ceil(num_samples/reduction_in_size))
    pattern_MR = np.zeros((fibres, new_num_samples))
    remainder = num_samples%reduction_in_size
    for n in np.arange(new_num_samples):
        if remainder != 0 and n == new_num_samples-1: # if last bin is shorter than 100 us
            pattern_MR[:,n] = np.sum(pattern[:,n*reduction_in_size:reduction_in_size*n+remainder],axis=1) 
        pattern_MR[:, n] = np.sum(pattern[:,n*reduction_in_size:reduction_in_size+reduction_in_size*n],axis=1)

    # windowing prep
    window_MR = np.hamming(length_MR)
    window_FT = np.hamming(length_FT)
    length_MR_downsampled = int(np.floor((new_num_samples-length_MR)/(length_MR/2) + 1))
    length_FT_downsampled = int(np.floor((num_samples-length_FT)/(length_FT/2) + 1))
    
    MR_downsampled = np.zeros((fibres, length_MR_downsampled))
    FT_downsampled = np.zeros((fibres, length_FT_downsampled))

    # convolve matrix per row with Hamming window
    for i in np.arange(length_FT_downsampled):
        window_i_FT = pattern[:, int(i*(length_FT/2)):int(i*(length_FT/2)+length_FT)]
        convolvedFT = np.sum((window_i_FT * window_FT), axis=1)
        FT_downsampled[:, i] = convolvedFT

        if i<length_MR_downsampled:
            window_i_MR = pattern[:, int(i*(length_MR/2)):int(i*(length_MR/2)+length_MR)]
            convolvedMR = np.sum((window_i_MR * window_MR), axis=1)
            MR_downsampled[:, i] = convolvedMR

    # scaling
    if scaling == 'Hines' or scaling==1:
        # scale between [0, 255]
        MR_downsampled = (MR_downsampled-MR_downsampled.min())/(MR_downsampled.max() - MR_downsampled.min())*255
        FT_downsampled = (FT_downsampled-FT_downsampled.min())/(FT_downsampled.max() - FT_downsampled.min())*255
    if scaling == 'Wirtzfeld' or scaling ==2:
        MR_downsampled = MR_downsampled/100e-6
        FT_downsampled = FT_downsampled/10e-6

    fiber_id_list = range(0, 3200, 10)
    # visualize MR
    if visualize_MR:
        plt.figure()
        x = np.linspace(0, new_num_samples*100e-6, length_MR_downsampled)
        y = np.array(fiber_id_list)*10
        plt.pcolormesh(x, y, MR_downsampled, cmap='inferno')
        cbar = plt.colorbar()
        cbar.set_label('Spikes')
        plt.ylabel('Apical                     Fiber number                   Basal')
        plt.xlabel('Time [s]')
        plt.title('Mean-rate neurogram (bin = 100us, Hamming=128)')
        plt.ylabel('Fibers')
    if visualize_FT:
        plt.figure()
        x = np.linspace(0, new_num_samples*10e-6, length_FT_downsampled)
        y = np.array(fiber_id_list)*10
        plt.pcolormesh(x, y, FT_downsampled, cmap='inferno')
        cbar = plt.colorbar()
        cbar.set_label('Spikes')
        plt.ylabel('Apical                     Fiber number                   Basal')
        plt.xlabel('Time [s]')
        plt.title('Fine-timing neurogram (bin = 10us, Hamming=32)')

    x=3
    return MR_downsampled, FT_downsampled


###############################
# NSIM local: https://en.wikipedia.org/wiki/Structural_similarity

def SSIM(patternR, patternD, alpha=1, beta=0, gamma=1, K1=0.01, K2=0.03):
    L = np.maximum(np.amax(patternR), np.amax(patternD))
    kernel_size = 3
    stride = 1 # according to https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
    # these values are from SSIM, are different 
    # in Tabiba: C1 = 6.5025,  C2 = C3 = 162.5625
    C1 = 6.5025 # (K1*L)**2
    C2 = 162.5625 # (K2*L)**2
    C3 = 162.5625 # C2/2

    (num_fibres, num_samples) = patternR.shape
    num_rows = int((num_fibres-kernel_size)/stride + 1)
    num_columns = int((num_samples-kernel_size)/stride + 1)

    S_all = []
    sk_all = []
    for row in np.arange(num_rows):
        row_end = row + kernel_size
        for column in np.arange(num_columns):
            column_end = column + kernel_size
            R = patternR[row:row_end, column:column_end]
            D = patternD[row:row_end, column:column_end]
            
            # luminance
            mu_R = np.mean(R)
            mu_D = np.mean(D)
            luminance = (2 * mu_R * mu_D + C1) / ((mu_R ** 2) + (mu_R**2) + C1)
            # contrast
            var_R = np.var(R)
            var_D = np.var(D)
            contrast = (2*var_R*var_D + C2) / ((var_R ** 2) + (var_D**2) + C2)
            # structure
            xy = []
            for n in np.arange(int(kernel_size**2)):
                xy.append((np.ravel(R)[n]-mu_R) * (np.ravel(D)[n]-mu_D)) # https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
            cov_RD = np.sum(xy)/(kernel_size**2-1)
            structure = (cov_RD + C3) / (var_R*var_D + C3)
            S_RD = (luminance**alpha) * (contrast**beta) * (structure **gamma)
            S_all.append(S_RD)
    return S_all

###############################
# NSIM overall


if __name__ == '__main__':   
    string1 = 'smrt_rate_5_depth_20_width_2p0_phase_0'
    string2 = 'smrt_rate_5_depth_20_width_1p4_phase_0'

    print('comparing', string1, 'and', string2)

    # load spiking patterns
    print('Loading spiking rates')
    pattern1 = load_spiking(string1)*10e-6 # currently in spikes/s --> spikes
    pattern2 = load_spiking(string2)*10e-6

    if pattern1.shape[0] != pattern2.shape[0]:
        raise ValueError('Not comparing same number of fibers!')
    
    if pattern1.shape[1] != pattern2.shape[1]:
        raise ValueError('Not comparing same bin size!')

    # preprocessing
    print('Pre-processing')
    MR_downsampled1, FT_downsampled1 = preprocessing(pattern1)
    MR_downsampled2, FT_downsampled2 = preprocessing(pattern2)

    from skimage.metrics import structural_similarity as ssim
    print('Using SSIM from sk-image')
    ssim_all = ssim(pattern1, pattern2)
    print('without pre-processing:', ssim_all)

    # Mean-rate
    ssim_MR = ssim(MR_downsampled1, MR_downsampled2)
    print('MR SSIM:', np.mean(ssim_MR))

    # Fine-timing
    ssim_FT = ssim(FT_downsampled1, FT_downsampled2)
    print('FT SSIM:', np.mean(ssim_FT))

    # NSIM
    print('Using own NSIM')
    S_all = SSIM(pattern1, pattern2)
    print('without pre-processing:', np.mean(S_all))

    # Mean-rate NSIM
    nsim_MR = SSIM(MR_downsampled1, MR_downsampled2)
    print('MR NSIM:', np.mean(nsim_MR))

    # Fine-timing
    nsim_FT = SSIM(FT_downsampled1, FT_downsampled2)
    print('FT NSIM:', np.mean(nsim_FT))

    NSIM = np.mean(S_all)
