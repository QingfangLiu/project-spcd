
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

TR = 1.5;

addpath(fullfile(pwd, 'MultivGLM_character'))
addpath(fullfile(pwd, 'MultivGLM_place'))

%%

% Loop over subjects
for subj = 1:length(subjectFolders)
    SubID = subjectFolders(subj).name;
    fprintf('\n\n=== Processing: %s ===\n', SubID);        
    run('run_trialwise_glms_place.m')
    run('run_trialwise_glms_char.m')
end