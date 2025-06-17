
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

dcm2niipath = '/Applications/MRIcroGL.app/Contents/Resources/dcm2niix';
size_threshold = 2 * 1024 * 1024; % Set size threshold (2 MB)
pattern_list = {'place', 'character', 'odor_scan', 'odorpred', 'premeal', 'postmeal'}; 


cfg.homeDir = '/Users/liuq13/SPCD_data';
cfg.niiDir = fullfile(cfg.homeDir, 'nii');
cfg.GLMDir = fullfile(cfg.HomeDir, 'GLM');
cfg.modelName = 'odor_scan';
cfg.subjects = dir(fullfile(cfg.niiDir, 'sub-*'));
cfg.nruns = 2;
cfg.TR = 1.5;
cfg.modelname = 'odor_scan';



% Get list of subject folders (those starting with 'sub-')
allEntries = dir(fullfile(niiDir, 'sub-*'));
subjectFolders = allEntries([allEntries.isdir]);  % keep only directories

subj = 1;
SubID = subjectFolders(subj).name;  % e.g., 'sub-01'
disp(['Processing: ', SubID]);

jobdir = fullfile(HomeDir, 'jobs', SubID);

