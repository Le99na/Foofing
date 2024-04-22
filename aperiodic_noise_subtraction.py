import numpy as np
import os
from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.gen import gen_aperiodic
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectra
from fooof.plts.annotate import plot_annotated_model
from fooof.plts.annotate import plot_annotated_peak_search
import h5py
import scipy
from scipy.signal import welch
import matplotlib.pyplot as plt



home_dat = os.path.dirname('/home/mc/Documents/MBA/')
end_dat = os.path.dirname('/media/mc/T7/PhD/Projekt2/analysis/_clean_data/tRNS-Paper/brainstorm/final_data_total/EC/bs_coherence_corrMats/complete/coh_difference_PrePost/')

os.chdir(home_dat)

# Load the .mat file with EEG data using h5py
with h5py.File('eeg_data10_2.mat', 'r') as file:
    # Get the EEG data from the HDF5 file
    eeg_data = file['EEG']['data'][:]  # Assuming 'data' is the field in your EEG structure

eeg = eeg_data.T

l_eeg = np.shape(eeg)[1]
plt_log = False


lst = []
for idx,chnl in enumerate(eeg):
    # Assuming you have preprocessed EEG data stored in 'eeg_data'
    os.chdir(home_dat)
    name = f'channel_{idx}'
    os.makedirs(name, exist_ok=True)
    os.chdir(f'{home_dat}/{name}')
    # Step 2: Compute the power spectrum of the preprocessed EEG data
    freqs, powers = scipy.signal.welch(chnl, fs=200)

    # Step 3: Fit a FOOOF model to the power spectrum
    fm = FOOOF(peak_width_limits=[1, 20], max_n_peaks=6, min_peak_height=0.15)
    fm.fit(freqs, powers,[3, 40])

    fm.plot(plt_log)
    plt.savefig(f'{name}_plot1.png')

    init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    init_flat_spec = fm.power_spectrum - init_ap_fit

    # Plot the initial aperiodic fit
    _, ax = plt.subplots(figsize=(12, 10))
    plot_spectra(fm.freqs, fm.power_spectrum, plt_log,
                 label='Original Power Spectrum', color='black', ax=ax)
    plot_spectra(fm.freqs, init_ap_fit, plt_log, label='Initial Aperiodic Fit',
                 color='blue', alpha=0.5, linestyle='dashed', ax=ax)
    plt.savefig(f'{name}_plot2.png')

    # Plot the flattened the power spectrum
    plot_spectra(fm.freqs, init_flat_spec, plt_log,
                 label='Flattened Spectrum', color='black')
    plt.savefig(f'{name}_plot3.png')

    plot_annotated_peak_search(fm)
    plt.savefig(f'{name}_plot4.png')

    # Plot the peak fit: created by re-fitting all of the candidate peaks together
    plot_spectra(fm.freqs, fm._peak_fit, plt_log, color='green', label='Final Periodic Fit')
    plt.savefig(f'{name}_plot5.png')

    # Plot the peak removed power spectrum, created by removing peak fit from original spectrum
    plot_spectra(fm.freqs, fm._spectrum_peak_rm, plt_log,
                 label='Peak Removed Spectrum', color='black')
    plt.savefig(f'{name}_plot6.png')

    # Plot the final aperiodic fit, calculated on the peak removed power spectrum
    _, ax = plt.subplots(figsize=(12, 10))
    plot_spectra(fm.freqs, fm._spectrum_peak_rm, plt_log,
                 label='Peak Removed Spectrum', color='black', ax=ax)
    plot_spectra(fm.freqs, fm._ap_fit, plt_log, label='Final Aperiodic Fit',
                 color='blue', alpha=0.5, linestyle='dashed', ax=ax)
    plt.savefig(f'{name}_plot7.png')

    final_flat_spec = fm.power_spectrum - fm._ap_fit

    lst.append(final_flat_spec)

array_data = np.array(lst)

freq = fm.freqs
mask = (freq >= 8) & (freq <= 12)

lst_all = []
for idx,i in enumerate(lst):
    alpha_pwr = np.mean(i[mask])
    lst_all.append(alpha_pwr)

# Specify the filename for the .mat file
file_name = 'foofed_data.mat'


import scipy.io
# Save the list into a .mat file
scipy.io.savemat(file_name, {'foofed_data': lst_all})

