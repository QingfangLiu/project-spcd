
% fix_sub14_premeal_cue_index_swap.m
%
% Swap values in the 'cue_index' column for sub-14 premeal event TSVs.
%
% TARGET FILES:
%   SPCD_data/nii/sub-14/func_PAID/premeal/
%   Pattern: sub-14_task-premeal_run-*_events.tsv
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

%% Set subject and folder (from your screenshot)
subID    = 'sub-14';
baseDir  = '/Users/liuq13/SPCD_data/nii';
taskName = 'premeal';

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

    % Skip older backups and cue-fix backups
    if endsWith(tsvPath, '_orig.tsv') || endsWith(tsvPath, '_beforeCueFix.tsv')
        fprintf('Skipping backup file: %s\n', tsvPath);
        continue;
    end

    % Create _beforeCueFix backup if it doesn't exist
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

    % Apply cue_index swap rules
    T.cue_index = swap_cue_index_values(T.cue_index);

    % Write back (overwrite current TSV)
    writetable(T, tsvPath, 'FileType', 'text', 'Delimiter', '\t');
    fprintf('Updated file: %s\n', tsvPath);
end

fprintf('Done.\n');

%% ---- helper function ----
function out = swap_cue_index_values(in)
% Robustly swap cue_index values:
%   1<->3, 2<->4, 7<->9, 8<->10
% Preserves NaNs / blanks / non-numeric strings unchanged.

    % Mapping pairs
    keys = [1 2 3 4 7 8 9 10];
    vals = [3 4 1 2 9 10 7 8];

    if isnumeric(in)
        x = in;
        out = x;

        % Treat values extremely close to integers as integers
        isFinite = isfinite(x);
        xr = round(x);

        % Only snap if very close to integer
        closeToInt = isFinite & (abs(x - xr) < 1e-6);

        % Apply mapping on snapped integer values
        for i = 1:numel(keys)
            mask = closeToInt & (xr == keys(i));
            out(mask) = vals(i);
        end

    elseif isstring(in) || iscellstr(in)
        s = string(in);
        out = s;  % default: unchanged

        % Convert to numeric where possible
        x = str2double(s);

        % Identify numeric entries (not NaN from conversion)
        canConvert = ~isnan(x);

        % Same "close to integer" logic
        xr = round(x);
        closeToInt = canConvert & isfinite(x) & (abs(x - xr) < 1e-6);

        % Apply mapping
        xOut = x;
        for i = 1:numel(keys)
            mask = closeToInt & (xr == keys(i));
            xOut(mask) = vals(i);
        end

        % Write back numeric-changed entries as strings; leave non-numeric as-is
        out(closeToInt) = string(xOut(closeToInt));

    else
        % Fallback: try to convert to string then proceed
        s = string(in);
        out = swap_cue_index_values(s);
    end
end
