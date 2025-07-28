% Trial-wise LSA GLM for char scan
% Each GLM estimates one trial with nuisance regressors and 2mm smoothed data

% ---- CONFIG ----
nruns = 4;
taskname = 'character';
glm_subfolder = 'character_trialwise_2mm';

% ---- LOOP OVER RUNS ----
for r = 1:nruns
    
    % Load event onsets
    tsv_file = fullfile(HomeDir, 'nii', SubID, 'func_PAID', taskname, ...
        sprintf('%s_task-%s_run-%d_events.tsv', SubID, taskname, r));
    event_table = readtable(tsv_file, 'FileType', 'text', 'Delimiter', '\t');
    
    onsets = event_table.onset;
    ntrials = height(event_table);

    % Locate 4D NIfTI file for this run
    scandir = fullfile(HomeDir, 'nii', SubID, 'func_PAID', taskname);
    nii_struct = dir(fullfile(scandir, sprintf('s2w2Scan1_%s_run_%d_*.nii', taskname, r)));
    if isempty(nii_struct)
        error('No s2w2 .nii file found for run %d in %s', r, scandir);
    end
    nii_file = fullfile(scandir, nii_struct(1).name);
    vols = spm_select('expand', nii_file);

    % Load nuisance regressors
    regfile = fullfile(HomeDir, 'NRegressor', SubID, ...
        sprintf('nuisance_regressors_%s_run_%d_zscored.txt', taskname, r));
    if ~exist(regfile, 'file')
        error('Nuisance regressor file not found: %s', regfile);
    end

    % ---- LOOP OVER TRIALS ----
    for t = 1:ntrials
        trial_onset = onsets(t);
        other_onsets = onsets;
        other_onsets(t) = [];

        % Output directory for this trial
        outdir = fullfile(GLMDir, glm_subfolder, SubID, sprintf('run%d_trial%02d', r, t));
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

        target_dir = fullfile(GLMDir, glm_subfolder, SubID);
        if ~exist(target_dir, 'dir'); mkdir(target_dir); end

        % Rename for clarity
        target_file = fullfile(target_dir, sprintf('beta_run%d_trial%02d.nii', r, t));
        
        % Move the beta file to subject-level folder
        movefile(beta_file, target_file);
        
        % Delete the rest of the GLM folder
        delete(fullfile(outdir, '*'));  % delete all files in trial folder
        rmdir(outdir);                  % remove now-empty trial folder

    end % end of trial
end % end of run
