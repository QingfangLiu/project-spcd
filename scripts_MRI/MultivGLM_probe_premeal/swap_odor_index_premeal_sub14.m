
% -------------------------------------------------------------------------
% swap_odor_index_premeal_sub14.m
%
% PURPOSE:
%   This script swaps the values in the 'odor_index' column for the subject
%   'sub-14' and task 'premeal', replacing:
%       1 -> 2
%       2 -> 1
%   All other values (0, NaN, etc.) are left unchanged.
%
% TARGET FILES:
%   1) Event files in:
%        SPCD_data/nii/sub-14/func_PAID/premeal/
%        Pattern: sub-14_task-premeal_run-*_events.tsv
%
%   2) Beta index files in:
%        SPCD_data/GLM/premeal_trialwise_2mm/sub-14/
%        Pattern: sub-14_premeal_run-*_beta_index.csv
%
% BACKUP:
%   Before overwriting, the script creates a backup copy in the same folder
%   with '_orig' appended to the filename (e.g., myfile_orig.tsv).
%   Backups are only created if they do not already exist.
%
% OUTPUT:
%   - The original file is overwritten with the updated odor_index values.
%   - Progress is printed to the MATLAB command window, including the count
%     of swapped values for each file.
%
% USAGE:
%   - This code was run after finishing bids events generator code and glm
%   code as a posthoc fix to sub-14
%
% -------------------------------------------------------------------------



% Set subject and folder
subID = 'sub-14';
baseDir = '/Users/liuq13/SPCD_data/nii'; 
taskName = 'premeal';

% Pattern to match the event TSV files
tsvPattern = fullfile(baseDir, subID, 'func_PAID', taskName, ...
    sprintf('%s_task-%s_run-*_events.tsv', subID, taskName));
tsvFiles = dir(tsvPattern);

for f = 1:length(tsvFiles)
    % Full path to file
    tsvPath = fullfile(tsvFiles(f).folder, tsvFiles(f).name);

    % Backup original
    [folderPath, baseName, ext] = fileparts(tsvPath);
    backupName = fullfile(folderPath, [baseName '_orig' ext]);
    if ~isfile(backupName)  % avoid overwriting existing backup
        copyfile(tsvPath, backupName);
        fprintf('Backup created: %s\n', backupName);
    else
        fprintf('Backup already exists: %s\n', backupName);
    end

    % Read the TSV file
    T = readtable(tsvPath, 'FileType', 'text', 'Delimiter', '\t');

    % Swap odor_index 1 <-> 2
    idx1 = T.odor_index == 1;
    idx2 = T.odor_index == 2;
    T.odor_index(idx1) = 2;
    T.odor_index(idx2) = 1;

    % Save updated table (same name as original)
    writetable(T, tsvPath, 'FileType', 'text', 'Delimiter', '\t');
    fprintf('Updated file: %s\n', tsvPath);
end

%%

% swap_glm_odor_index_sub14.m
% Swap odor_index 1 <-> 2 in:
%   SPCD_data/GLM/premeal_trialwise_2mm/sub-14/*_beta_index.csv

rootDir = '/Users/liuq13/SPCD_data';   
glmDir = fullfile(rootDir, 'GLM', 'premeal_trialwise_2mm', subID);
glmPat = fullfile(glmDir, sprintf('%s_premeal_run-*_beta_index.csv', subID));
glmFiles = dir(glmPat);

for k = 1:numel(glmFiles)
    fpath = fullfile(glmFiles(k).folder, glmFiles(k).name);

    % Read CSV
    T = readtable(fpath);

    if ~ismember('odor_index', T.Properties.VariableNames)
        warning('No odor_index in %s (skipped).', glmFiles(k).name);
        continue
    end

    % Backup original once
    [p,b,e] = fileparts(fpath);
    backup = fullfile(p, [b '_orig' e]);
    if ~isfile(backup)
        copyfile(fpath, backup);
        fprintf('Backup created: %s\n', backup);
    else
        fprintf('Backup already exists: %s\n', backup);
    end

    % Swap 1 <-> 2
    one = T.odor_index == 1;
    two = T.odor_index == 2;
    T.odor_index(one) = 2;
    T.odor_index(two) = 1;

    % Write back (CSV)
    writetable(T, fpath);
    fprintf('[GLM] %s | #1->2: %d, #2->1: %d\n', glmFiles(k).name, nnz(one), nnz(two));
end

