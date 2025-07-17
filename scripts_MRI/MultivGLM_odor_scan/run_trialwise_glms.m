% Trial-wise LSA GLM for odor scan decoding
% Each GLM estimates one trial with nuisance regressors and 2mm smoothed data

% ---- CONFIG ----
nruns = 2;

% Timing
blank_dur = 10;
white_cross_dur = 2;
blue_cross_dur = 2;

% ---- LOOP OVER RUNS ----
for r = 1:nruns
    % Load task file
    design_file = sprintf('IndepScanOdorConds%d.xlsx', r);
    design_path = fullfile(HomeDir, 'beh_raw', SubID, design_file);
    task_design = readtable(design_path);
    ITIs = task_design.ITIDur;
    ntrials = height(task_design);

    % Compute sniff cue onsets
    sniff_onsets = nan(1, ntrials);
    sniff_onsets(1) = blank_dur + white_cross_dur;
    for t = 2:ntrials
        sniff_onsets(t) = sniff_onsets(t-1) + white_cross_dur + blue_cross_dur + ITIs(t-1);
    end

    % Locate 4D NIfTI file for this run
    scandir = fullfile(HomeDir, 'nii', SubID, 'func_PAID', 'odor_scan');
    nii_struct = dir(fullfile(scandir, sprintf('s2w2Scan1_odor_scan_run_%d_*.nii', r)));
    if isempty(nii_struct)
        error('No s2w2 .nii file found for run %d in %s', r, scandir);
    elseif length(nii_struct) > 1
        warning('Multiple NIfTI files found for run %d, using the first one.', r);
    end
    nii_file = fullfile(scandir, nii_struct(1).name);
    vols = spm_select('expand', nii_file);  % expand to 3D

    % Load nuisance regressors
    regfile = fullfile(HomeDir, 'NRegressor', SubID, ...
        sprintf('nuisance_regressors_odor_scan_run_%d_zscored.txt', r));
    if ~exist(regfile, 'file')
        error('Nuisance regressor file not found: %s', regfile);
    end

    % ---- LOOP OVER TRIALS ----
    for t = 1:ntrials
        trial_onset = sniff_onsets(t);
        other_onsets = sniff_onsets;
        other_onsets(t) = [];

        % Output directory for this trial
        outdir = fullfile(GLMDir, 'odor_scan_trialwise_2mm', SubID, sprintf('run%d_trial%02d', r, t));
        if ~exist(outdir, 'dir'), mkdir(outdir); end

        % Build SPM job
        spm_jobman('initcfg');
        matlabbatch = {};

        % Model specification
        matlabbatch{1}.spm.stats.fmri_spec.dir = {outdir};
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
        matlabbatch{1}.spm.stats.fmri_spec.sess.scans = cellstr(vols);
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.1;  
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';

        % Regressor 1: trial of interest
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name = 'trial_of_interest';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset = trial_onset;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = 0;

        % Regressor 2: all other trials
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name = 'other_trials';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset = other_onsets';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = 0;

        % Nuisance regressors
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {regfile};
        matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;

        % Estimation
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(outdir, 'SPM.mat')};

        % Run
        spm_jobman('run', matlabbatch);

        
        % Define source and target
        beta_file = fullfile(outdir, 'beta_0001.nii');
        target_dir = fullfile(GLMDir, 'odor_scan_trialwise_2mm', SubID);
        
        % Rename for clarity
        target_file = fullfile(target_dir, sprintf('beta_run%d_trial%02d.nii', r, t));
        
        % Move the beta file to subject-level folder
        movefile(beta_file, target_file);
        
        % Delete the rest of the GLM folder
        delete(fullfile(outdir, '*'));  % delete all files in trial folder
        rmdir(outdir);                  % remove now-empty trial folder


    end % end of trial
end % end of run
