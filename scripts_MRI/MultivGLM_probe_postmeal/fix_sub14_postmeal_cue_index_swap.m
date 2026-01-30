% fix_sub14_postmeal_cue_index_swap.m
%
% Swap values in the 'cue_index' column for sub-14 postmeal event TSVs.
%
% TARGET FILES:
%   SPCD_data/nii/sub-14/func_PAID/postmeal/
%   Pattern: sub-14_task-postmeal_run-*_events.tsv
%
% SWAP RULES (cue_index only):
%   1 <-> 3
%   2 <-> 4
%   7 <-> 9
%   8 <-> 10
% All other values (0, NaN, blanks, etc.) unchanged.
%
% BACKUP POLICY:
%   - Keep existing *_orig.tsv untouched.
%   - Before overwriting current TSV, create *_beforeCueFix.tsv if it doesn't exist.
%   - Do NOT overwrite an existing *_beforeCueFix.tsv.

clear; clc;

%% Set subject and folder
subID    = 'sub-14';
baseDir  = '/Users/liuq13/SPCD_data/nii';
taskName = 'postmeal';

%% Pattern to match TSV event files
tsvPattern = fullfile(baseDir, subID, 'func_PAID', taskName, ...
    sprintf('%s_task-%s_run-*_events.tsv', subID, taskName));

tsvFiles = dir(tsvPattern);

if isempty(tsvFiles)
    fprintf('No files found for pattern:\n  %s\n', tsvPattern);
    return;
end

%% Process each TSV
for f = 1:numel(tsvFiles)
    tsvPath = fullfile(tsvFiles(f).folder, tsvFiles(f).name);

    % Skip older backups
    if endsWith(tsvPath, '_orig.tsv') || endsWith(tsvPath, '_beforeCueFix.tsv')
        fprintf('Skipping backup file: %s\n', tsvPath);
        continue;
    end

    % Create _beforeCueFix backup if needed
    [folderPath, baseName, ext] = fileparts(tsvPath);
    beforeFixBackup = fullfile(folderPath, [baseName '_beforeCueFix' ext]);

    if ~isfile(beforeFixBackup)
        copyfile(tsvPath, beforeFixBackup);
        fprintf('Backup created: %s\n', beforeFixBackup);
    else
        fprintf('Backup already exists: %s\n', beforeFixBackup);
    end

    % Read TSV
    T = readtable(tsvPath, 'FileType', 'text', 'Delimiter', '\t', 'TextType', 'string');

    % Sanity check
    if ~ismember('cue_index', T.Properties.VariableNames)
        warning('No cue_index column found in %s (skipping).', tsvPath);
        continue;
    end

    % Swap cue_index values
    T.cue_index = swap_cue_index_values(T.cue_index);

    % Overwrite TSV
    writetable(T, tsvPath, 'FileType', 'text', 'Delimiter', '\t');
    fprintf('Updated file: %s\n', tsvPath);
end

fprintf('Done.\n');

%% ---- helper function ----
function out = swap_cue_index_values(in)
% Robust cue_index swapping:
%   1<->3, 2<->4, 7<->9, 8<->10

    keys = [1 2 3 4 7 8 9 10];
    vals = [3 4 1 2 9 10 7 8];

    if isnumeric(in)
        x = in;
        out = x;

        isFinite = isfinite(x);
        xr = round(x);
        closeToInt = isFinite & (abs(x - xr) < 1e-6);

        for i = 1:numel(keys)
            mask = closeToInt & (xr == keys(i));
            out(mask) = vals(i);
        end

    elseif isstring(in) || iscellstr(in)
        s = string(in);
        out = s;

        x = str2double(s);
        canConvert = ~isnan(x);
        xr = round(x);
        closeToInt = canConvert & isfinite(x) & (abs(x - xr) < 1e-6);

        xOut = x;
        for i = 1:numel(keys)
            mask = closeToInt & (xr == keys(i));
            xOut(mask) = vals(i);
        end

        out(closeToInt) = string(xOut(closeToInt));

    else
        out = in;
    end
end
