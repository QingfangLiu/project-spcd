
clear; clc;

addpath('/Users/liuq13/Applications/spm12/')
addpath('/Users/liuq13/xjview10/xjview/')

%
HomeDir = '/Users/liuq13/SPCD_data';
niiDir = fullfile(HomeDir, 'nii');
GLMDir = fullfile(HomeDir, 'GLM');

% Get list of subject folders (those starting with 'sub-')
allEntries = dir(fullfile(niiDir, 'sub-*'));
subjectFolders = allEntries([allEntries.isdir]);  % keep only directories


TEs = [12.60, 29.16, 45.72, 62.28];
nechos = 4;
TR = 1.5;

dcm2niipath = '/Applications/MRIcroGL.app/Contents/Resources/dcm2niix';
size_threshold = 2 * 1024 * 1024; % Set size threshold (2 MB)
pattern_list = {'place', 'character', 'odor_scan', 'odorpred', 'premeal', 'postmeal'}; 


addpath(fullfile(pwd, 'UnivGLM'));

for subj = 1:length(subjectFolders)
    SubID = subjectFolders(subj).name;
    fprintf('\n\n=== Processing: %s ===\n', SubID);

    try
        % Set up paths
        jobdir = fullfile(HomeDir, 'jobs', SubID);
        if ~exist(jobdir, 'dir'); mkdir(jobdir); end

        modelname = 'odor_scan';
        modeldir = fullfile(GLMDir, modelname, SubID);
        spmmat_path = fullfile(modeldir, 'SPM.mat');

        % === Check if GLM has been estimated ===
        if exist(spmmat_path, 'file'  )
            fprintf('⚠️  GLM already exists for %s — skipping.\n', SubID);
            continue;
        end

        % Step 1: Find and save onsets
        fprintf('\n--- Running: find_onsets_odor_scan ---\n');
        find_onsets_odor_scan;

        % Step 2: Specify and estimate GLM
        fprintf('\n--- Running: specify_est_odor_scan ---\n');
        specify_est_odor_scan;

        % Step 3: Add contrasts
        fprintf('\n--- Running: add_contrast_odor_scan ---\n');
        add_contrast_odor_scan;

        fprintf('\n✅ All steps complete for %s\n', SubID);

    catch ME
        fprintf('\n❌ Skipping %s due to error: %s\n', SubID, ME.message);

        % Optional: Log error
        logfile = fullfile(HomeDir, 'error_log.txt');
        fid = fopen(logfile, 'a');
        fprintf(fid, '[%s] Subject %s failed: %s\n', datestr(now), SubID, ME.message);
        fclose(fid);

        continue;
    end
end




