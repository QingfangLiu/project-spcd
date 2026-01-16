% -------------------------------------------------------------------------
% script adapted from premeal 
% -------------------------------------------------------------------------

clear; clc;

subID   = 'sub-14';
rootDir = '/Users/liuq13/SPCD_data';

glmDir  = fullfile(rootDir, 'GLM', 'postmeal_trialwise_2mm', subID);
glmPat  = fullfile(glmDir, sprintf('%s_postmeal_run-*_beta_index.csv', subID));
glmFiles = dir(glmPat);

if isempty(glmFiles)
    error('No files found matching: %s', glmPat);
end

for k = 1:numel(glmFiles)
    fpath = fullfile(glmFiles(k).folder, glmFiles(k).name);

    % Read CSV
    T = readtable(fpath);

    if ~ismember('cue_index', T.Properties.VariableNames)
        warning('No cue_index in %s (skipped).', glmFiles(k).name);
        continue
    end

    % Backup current working file once: *_beforeCueFix.csv
    [p,b,e] = fileparts(fpath);
    backup = fullfile(p, [b '_beforeCueFix' e]);
    if ~isfile(backup)
        copyfile(fpath, backup);
        fprintf('Backup created: %s\n', backup);
    else
        fprintf('Backup already exists: %s\n', backup);
    end

    % ---- Swap cue_index using the same "mask then assign" style ----
    ci = T.cue_index;

    % If cue_index came in as text, attempt conversion
    if iscell(ci) || isstring(ci) || ischar(ci)
        ci_num = str2double(string(ci));
        if any(isnan(ci_num) & ~ismissing(string(ci)))
            error('cue_index contains non-numeric values in %s', glmFiles(k).name);
        end
        ci = ci_num;
    end

    % Masks based on ORIGINAL values (important for correct swaps)
    idx1  = (ci == 1);
    idx3  = (ci == 3);
    idx2  = (ci == 2);
    idx4  = (ci == 4);
    idx7  = (ci == 7);
    idx9  = (ci == 9);
    idx8  = (ci == 8);
    idx10 = (ci == 10);

    % Apply swaps
    ci(idx1)  = 3;
    ci(idx3)  = 1;

    ci(idx2)  = 4;
    ci(idx4)  = 2;

    ci(idx7)  = 9;
    ci(idx9)  = 7;

    ci(idx8)  = 10;
    ci(idx10) = 8;

    T.cue_index = ci;

    % Write back (CSV) overwriting original
    writetable(T, fpath);

    fprintf('[CueFix] %s | 1<->3: (%d,%d)  2<->4: (%d,%d)  7<->9: (%d,%d)  8<->10: (%d,%d)\n', ...
        glmFiles(k).name, ...
        nnz(idx1), nnz(idx3), ...
        nnz(idx2), nnz(idx4), ...
        nnz(idx7), nnz(idx9), ...
        nnz(idx8), nnz(idx10));
end

fprintf('\nDone. Current CSVs backed up as *_beforeCueFix.csv and overwritten with cue_index swaps.\n');
