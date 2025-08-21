
% -------------------------------------------------------------------------
% swap_odor_num_odorpred_sub14.m
%
% PURPOSE:
%   For subject 'sub-14' and task 'odorpred', swap the values of the
%   odor label column so that:
%       1 -> 2
%       2 -> 1
%   Values 0, NaN, or 'n/a' are left unchanged.
%
% TARGET FILES:
%   SPCD_data/nii/sub-14/func_PAID/odorpred/
%   Pattern: sub-14_task-odorpred_run-*_events.tsv
%
% BACKUP:
%   Creates a sibling backup with '_orig' appended (only if not present).
%
% NOTES:
%   - The odorpred events use column name 'odor_num' (string 'n/a' for
%     non-odor rows). This script handles string or numeric storage.
%   - Only events with numeric odor labels are affected; others remain.
% -------------------------------------------------------------------------

clear; clc;

subID    = 'sub-14';
baseDir  = '/Users/liuq13/SPCD_data/nii';
taskName = 'odorpred';

% Locate event TSVs
tsvPattern = fullfile(baseDir, subID, 'func_PAID', taskName, ...
    sprintf('%s_task-%s_run-*_events.tsv', subID, taskName));
tsvFiles = dir(tsvPattern);

if isempty(tsvFiles)
    fprintf('No events.tsv found matching: %s\n', tsvPattern);
end

for f = 1:numel(tsvFiles)
    tsvPath = fullfile(tsvFiles(f).folder, tsvFiles(f).name);

    % Backup original once
    [folderPath, baseName, ext] = fileparts(tsvPath);
    backupPath = fullfile(folderPath, [baseName '_orig' ext]);
    if ~isfile(backupPath)
        copyfile(tsvPath, backupPath);
        fprintf('Backup created: %s\n', backupPath);
    else
        fprintf('Backup already exists: %s\n', backupPath);
    end

    % Read
    T = readtable(tsvPath, 'FileType','text', 'Delimiter','\t', 'VariableNamingRule','preserve');

    % Find odor column (prefer 'odor_num', fallback to 'odor_index' if present)
    odorColName = '';
    if any(strcmp(T.Properties.VariableNames, 'odor_num'))
        odorColName = 'odor_num';
    elseif any(strcmp(T.Properties.VariableNames, 'odor_index'))
        odorColName = 'odor_index';
    else
        fprintf('⚠ Skipping (no odor column) : %s\n', tsvPath);
        continue
    end

    % Convert odor column to string for safe editing (handles 'n/a' / numbers)
    od_str = string(T.(odorColName));
    od_num = str2double(od_str);          % numeric where possible, NaN otherwise

    % Identify swaps
    idx1 = (od_num == 1);
    idx2 = (od_num == 2);
    n1 = sum(idx1);
    n2 = sum(idx2);

    % Apply swap (write back as strings to preserve 'n/a' style)
    od_str(idx1) = "2";
    od_str(idx2) = "1";
    T.(odorColName) = cellstr(od_str);    % keep as char cell for TSV friendliness

    % Save
    writetable(T, tsvPath, 'FileType','text', 'Delimiter','\t');
    fprintf('Updated: %s | swapped 1->2: %d, 2->1: %d\n', tsvPath, n1, n2);
end
