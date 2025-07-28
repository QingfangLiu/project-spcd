

clear; clc;

addpath('/Users/liuq13/Applications/spm12/')
addpath('/Users/liuq13/xjview10/xjview/')

%
HomeDir = '/Users/liuq13/SPCD_data';
niiDir = fullfile(HomeDir, 'nii');
GLMDir = fullfile(HomeDir, 'GLM');

maskPath = fullfile(HomeDir, 'masks', 'gm_0.1.nii'); 

% Get list of subject folders (those starting with 'sub-')
allEntries = dir(fullfile(niiDir, 'sub-*'));
subjectFolders = allEntries([allEntries.isdir]);  % keep only directories

%% Group-Level Analysis: Odor > Clean Air

% === Settings ===
clear matlabbatch
modelname = 'odor_scan';
contrast_name = 'odor>clean-air';
group_model_dir = fullfile(GLMDir, modelname, 'group');

if ~exist(group_model_dir, 'dir')
    mkdir(group_model_dir);
end

% === Collect contrast images ===
con_images = {};  % full paths to con_0001.nii files

for subj = 1:length(subjectFolders)
    SubID = subjectFolders(subj).name;
    con_file = fullfile(GLMDir, modelname, SubID, 'con_0001.nii');  % first contrast

    if exist(con_file, 'file')
        con_images{end+1, 1} = con_file;
    else
        fprintf('⚠️  Missing contrast for %s — skipping.\n', SubID);
    end
end

% === Design Specification: One-sample t-test ===
matlabbatch{1}.spm.stats.factorial_design.dir = {group_model_dir};
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = con_images;
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {maskPath};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

% === Model Estimation ===
matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(group_model_dir, 'SPM.mat')};
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

% === Contrast: Odor > Clean Air ===
matlabbatch{3}.spm.stats.con.spmmat = {fullfile(group_model_dir, 'SPM.mat')};
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = contrast_name;
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;  % one-sample t-test
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.delete = 1;

% === Run ===
fprintf('\nRunning group-level GLM for contrast: %s\n', contrast_name);
spm('Defaults','fMRI');
spm_jobman('initcfg');
spm_jobman('run', matlabbatch);
