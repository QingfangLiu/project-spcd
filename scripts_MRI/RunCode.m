
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

log_file = fullfile(GLMDir, 'error_log.txt');

% Start fresh log
if exist(log_file, 'file')
    delete(log_file);  % remove old log if exists
end

% Optional in-memory log
error_log = {};
log_index = 1;

addpath(fullfile(pwd, 'MultivGLM_character'))
addpath(fullfile(pwd, 'MultivGLM_place'))

%%

% Loop over subjects
for subj = 1:length(subjectFolders)
    SubID = subjectFolders(subj).name;
    fprintf('\n\n=== Processing: %s ===\n', SubID);

    try
        
        run('run_trialwise_glms_place.m')
        run('run_trialwise_glms_char.m')

    catch ME
        % Print warning
        warning('Error processing %s: %s', SubID, ME.message);

        % Immediate write to log file
        fid = fopen(log_file, 'a');  % append mode
        if fid ~= -1
            fprintf(fid, '[%s] Subject: %s\nError: %s\n\n', ...
                datestr(now), SubID, ME.message);
            fclose(fid);
        else
            warning('Could not open log file to write error for %s', SubID);
        end

        % Also save in memory (optional)
        error_log{log_index,1} = SubID;
        error_log{log_index,2} = ME.message;
        log_index = log_index + 1;
    end
end