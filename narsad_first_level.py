import bids
from bids import BIDSLayout, BIDSValidator
from nilearn.image import load_img

from scipy.stats import norm
import pandas as pd
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.image import mean_img
from nilearn import plotting
import nibabel as nib
from nilearn import image as nimg

def load_bids_events(layout,onsets, subject, task, phaseno):
    '''Create a design_matrix instance from BIDS event file'''
    
    tr = layout.get_tr()
    # change lines below -- can change to "mask", change task to "self-other"
    func_files = layout.get(subject=subject,
                        datatype='func', task=task,
                        desc='preproc',
                        space='MNI152NLin2009cAsym',
                        extension='nii.gz',
                       return_type='file')
    func_file = nimg.load_img(func_files)
    n_tr = func_file.shape[-1]
    onsets=onsets[phaseno]
    #onsets = pd.read_csv(layout_raw.get(subject=subject, suffix='events')[run].path, sep='\t') -- line to use if events changes by 
    
    # line below is isolating the onset, duration, and trial type columns -- change according to events.tsv format 
    onsets_actual = onsets.iloc[:, [0,1,3]]
    onsets_actual.columns = ['onset', 'duration','trial_type'] # make sure this order matches with what's loaded in as "onsets_actua
    sampling_freq = 1/tr
    n_scans=n_tr
    return onsets_actual, tr, n_scans


# Function to create first-level design matrices and fit the GLM
def create_and_fit_glm(layout, subjects, onsets, output_dir):
    p001_unc = norm.isf(0.001)

    file_lists_phase2 = {"CSR-CSS": list(), "CSR-CS-": list(), "CSS-CS-": list(), "CSR-fix": list(), 
                         "CSS-fix": list(), "CSminus-fix": list()}
    file_lists_phase3 = {"CSR-CSS": list(), "CSR-CS-": list(), "CSS-CS-": list(), "CSR-fix": list(), 
                         "CSS-fix": list(), "CSminus-fix": list()}
    tasks = ['phase2', 'phase3']

    for phaseno, task in enumerate(tasks):
        for sub in subjects:
            fmri_imgs = layout.get(subject=sub,
                                   datatype='func', task=task,
                                   desc='preproc',
                                   space='MNI152NLin2009cAsym',
                                   extension='nii.gz',
                                   return_type='file')

            hrf_model = "spm"  # canonical hrf
            high_pass = 0.01  # The cutoff for the drift model is 0.01 Hz.

            confound_files = layout.get(subject=sub,
                                        datatype='func', task=task,
                                        desc='confounds',
                                        extension="tsv",
                                        return_type='file')

            confound_vars = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
                             'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
                             'trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
                             'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2',
                             'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2',
                             'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2',
                             'csf', 'csf_derivative1', 'csf_derivative1_power2', 'csf_power2',
                             'white_matter', 'white_matter_derivative1', 'white_matter_derivative1_power2', 'white_matter_power2']

            design_matrices = []
            print(f"Creating First Level Design matrix for subject {sub}, task {task}...")

            for img in enumerate(fmri_imgs):
                events, tr, n_scans = load_bids_events(layout, onsets, sub, task, phaseno)
                frame_times = np.arange(n_scans) * tr

                confound_file = confound_files[0]
                confound_df = pd.read_csv(confound_file, delimiter='\t')
                confound_df = confound_df[confound_vars]
                confound_df.fillna(0, inplace=True)

                design_matrix = make_first_level_design_matrix(
                    frame_times,
                    events,
                    hrf_model=hrf_model,
                    drift_model="polynomial",
                    drift_order=3,
                    add_regs=confound_df,
                    add_reg_names=confound_vars,
                    high_pass=high_pass,
                )

                design_matrices.append(design_matrix)

            contrast_matrix = np.eye(design_matrix.shape[1])
            basic_contrasts = {
                column: contrast_matrix[i]
                for i, column in enumerate(design_matrix.columns)
            }

            contrasts = {
                "CSR-CSS": (basic_contrasts["CSR"] - basic_contrasts["CSS"]),
                "CSR-CS-": (basic_contrasts["CSR"] - basic_contrasts["CS-"]),
                "CSS-CS-": (basic_contrasts["CSS"] - basic_contrasts["CS-"]),
                "CSR-fix": (basic_contrasts["CSR"] - basic_contrasts["FIXATION"]),
                "CSS-fix": (basic_contrasts["CSS"] - basic_contrasts["FIXATION"]),
                "CSminus-fix": (basic_contrasts["CS-"] - basic_contrasts["FIXATION"]),                
            }

            fmri_glm = FirstLevelModel().fit(fmri_imgs, design_matrices=design_matrices)

            for contrast_id, contrast_val in contrasts.items():
                outputs = fmri_glm.compute_contrast(contrast_val, output_type='all')
                fname = f"{output_dir}/first_level_results/{sub}_{contrast_id}_{task}.nii.gz"
                zname = f"{output_dir}/first_level_results/plotting/{sub}_{contrast_id}_{task}.nii.gz"
                nib.save(outputs['effect_size'], fname)
                nib.save(outputs['z_score'], zname)
                plotting.plot_glass_brain(
                    outputs['z_score'],
                    threshold=p001_unc,
                    title=contrast_id,
                    colorbar=True,
                    plot_abs=False,
                    display_mode="z",
                )
                plotting.show()

                if phaseno == 0:
                    file_lists_phase2[contrast_id].append(fname)
                else:
                    file_lists_phase3[contrast_id].append(fname)

    return file_lists_phase2, file_lists_phase3
