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
from scipy.io import savemat
import pandas as pd



home_dat = os.path.dirname('E:/Foofing/mat-files/')
end_dat = os.path.dirname('E:/Foofing/mat-files/figures_knee/')
annotated = os.path.dirname('E:/Foofing/mat-files/figures_knee_annotated/')
aperiodic = os.path.dirname('E:/Foofing/mat-files/aperiodic_noise_adjustment/')

os.chdir(home_dat)

fls = os.listdir()
fls = [i for i in fls if i.__contains__('.mat')]

#testing for a knee in the data

for idx,i in enumerate(fls):
    # Load the .mat file
    nam = fls[idx]
    cond = nam.split('_')[-1]

    mat = scipy.io.loadmat(i)

    # Access the EEG struct
    if cond == 'corr.mat':
        EEG = mat['EEG_corr']
    else:
        EEG = mat['EEG_icorr']

    ch_av = np.mean(EEG,0)
    cond_av = np.mean(ch_av,1)

    freqs, powers = scipy.signal.welch(cond_av, fs=200)

    fm = FOOOF(peak_width_limits=[1, 20], max_n_peaks=6, min_peak_height=0.15)
    fm.fit(freqs, powers, [3, 40])

    # plt.plot(np.log(freqs[3:40]), np.log(powers[3:40]))
    # plt.xlabel('log(Frequency)')
    # plt.ylabel('log(Power)')
    # plt.savefig(f'{end_dat}/{nam[:-4]}_plot_knee.png')
    # plt.close()

    plot_annotated_model(fm, plt_log=True)
    plt.savefig(f'{annotated}/{nam[:-4]}_plot_model.png')
    plt.close()


l_eeg = np.shape(eeg)[1]
plt_log = False


lst = []
for idx,i in enumerate(fls):
    # Assuming you have preprocessed EEG data stored in 'eeg_data'
    nam = fls[idx][:-4]
    cond = nam.split('_')[-1]

    os.makedirs(f'{aperiodic}/{nam}', exist_ok=True)

    mat = scipy.io.loadmat(i)

    # Access the EEG struct
    if cond == 'corr':
        EEG = mat['EEG_corr']
    else:
        EEG = mat['EEG_icorr']

    ch_av = np.mean(EEG, 0)
    cond_av = np.mean(ch_av, 1)

    # Step 2: Compute the power spectrum of the preprocessed EEG data
    freqs, powers = scipy.signal.welch(cond_av, fs=200)

    # Step 3: Fit a FOOOF model to the power spectrum
    fm = FOOOF(peak_width_limits=[1, 10], max_n_peaks=6, min_peak_height=abs(0.1))
    fm.fit(freqs, powers,[3, 40])

    fm.plot(plt_log)
    plt.savefig(f'{aperiodic}/{nam}/plot1.png')
    plt.close()

    init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    init_flat_spec = fm.power_spectrum - init_ap_fit

    # Plot the initial aperiodic fit
    _, ax = plt.subplots(figsize=(12, 10))
    plot_spectra(fm.freqs, fm.power_spectrum, plt_log,
                 label='Original Power Spectrum', color='black', ax=ax)
    plot_spectra(fm.freqs, init_ap_fit, plt_log, label='Initial Aperiodic Fit',
                 color='blue', alpha=0.5, linestyle='dashed', ax=ax)
    plt.savefig(f'{aperiodic}/{nam}/plot2.png')
    plt.close()

    # Plot the flattened the power spectrum
    plot_spectra(fm.freqs, init_flat_spec, plt_log,
                 label='Flattened Spectrum', color='black')
    plt.savefig(f'{aperiodic}/{nam}/plot3.png')
    plt.close()

    plot_annotated_peak_search(fm)
    for k in plt.get_fignums():
        fig = plt.figure(k)
        fig.savefig(f'{aperiodic}/{nam}/plot4_{k}.png')
        plt.close(fig)

    # Plot the peak fit: created by re-fitting all of the candidate peaks together
    plot_spectra(fm.freqs, fm._peak_fit, plt_log, color='green', label='Final Periodic Fit')
    plt.savefig(f'{aperiodic}/{nam}/plot5.png')
    plt.close()

    # Plot the peak removed power spectrum, created by removing peak fit from original spectrum
    plot_spectra(fm.freqs, fm._spectrum_peak_rm, plt_log,
                 label='Peak Removed Spectrum', color='black')
    plt.savefig(f'{aperiodic}/{nam}/plot6.png')
    plt.close()

    # Plot the final aperiodic fit, calculated on the peak removed power spectrum
    _, ax = plt.subplots(figsize=(12, 10))
    plot_spectra(fm.freqs, fm._spectrum_peak_rm, plt_log,
                 label='Peak Removed Spectrum', color='black', ax=ax)
    plot_spectra(fm.freqs, fm._ap_fit, plt_log, label='Final Aperiodic Fit',
                 color='blue', alpha=0.5, linestyle='dashed', ax=ax)
    plt.savefig(f'{aperiodic}/{nam}/plot7.png')
    plt.close()

    final_flat_spec = fm.power_spectrum - fm._ap_fit

    # Define the filename for the .mat file
    filename = f'{nam}_flat.mat'

    # Create a dictionary to store the array with a variable name
    mat_dict = {'array_data': final_flat_spec}

    # Save the ndarray to a .mat file
    savemat(f'{aperiodic}/{nam}/{filename}', mat_dict)

    lst.append(final_flat_spec)

freq = fm.freqs
mask = (freq >= 8) & (freq <= 12)

lst_all = {}
for idx,i in enumerate(lst):
    alpha_pwr = np.mean(i[mask])
    lst_all[str(fls[idx][:-4])] = float(alpha_pwr)
    print(fls[idx][:-4])

df = pd.DataFrame(list(lst_all.items()), columns=['Keyword', 'Value'])

df.to_excel('average_alpha_defoofed.xlsx')


