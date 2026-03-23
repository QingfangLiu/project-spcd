
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
%   Event files in:
%        SPCD_data/nii/sub-14/func_PAID/premeal/
%        Pattern: sub-14_task-premeal_run-*_events.tsv
%
% BACKUP:
%   Before overwriting, the script creates a backup copy in the same folder
%   with '_orig' appended to the filename (e.g., myfile_orig.tsv).
%   Backups are only created if they do not already exist.
%
% OUTPUT:
%   - The original file is overwritten with the updated odor_index values.
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