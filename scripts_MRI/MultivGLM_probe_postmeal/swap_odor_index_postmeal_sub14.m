
% -------------------------------------------------------------------------
% PURPOSE
%   Swap 'odor_index' values (1 <-> 2) for sub-14 in a given task
%   across BOTH:
%     1) nii/.../<task>/sub-14_task-<task>_run-*_events.tsv
%     2) GLM/<task>_trialwise_2mm/sub-14/sub-14_<task>_run-*_beta_index.csv
%   Other values (0, NaN, etc.) are unchanged. Creates *_orig backups.
%
% CONFIG: set paths and task below.
% -------------------------------------------------------------------------

%% --- CONFIG ---
subID    = 'sub-14';
taskName = 'postmeal';  

rootDir  = '/Users/liuq13/SPCD_data';
niiDir   = fullfile(rootDir, 'nii');
glmDir   = fullfile(rootDir, 'GLM', sprintf('%s_trialwise_2mm', taskName), subID);

%% --- 1) NII events: swap in .../nii/sub-14/func_PAID/<task>/*.tsv
eventsPat = fullfile(niiDir, subID, 'func_PAID', taskName, ...
    sprintf('%s_task-%s_run-*_events.tsv', subID, taskName));
evFiles = dir(eventsPat);

for k = 1:numel(evFiles)
    fpath = fullfile(evFiles(k).folder, evFiles(k).name);
    T = readtable(fpath, 'FileType','text', 'Delimiter','\t');

    if ~ismember('odor_index', T.Properties.VariableNames)
        warning('[EVENTS] %s: no odor_index (skipped).', evFiles(k).name);
        continue
    end

    % swap 1 <-> 2
    one = T.odor_index == 1;
    two = T.odor_index == 2;
    T.odor_index(one) = 2;
    T.odor_index(two) = 1;

    % backup + write
    backup_and_write(T, fpath, '\t');
    fprintf('[EVENTS] %s | #1->2: %d, #2->1: %d\n', evFiles(k).name, nnz(one), nnz(two));
end

%% --- 2) GLM beta_index: swap in .../GLM/<task>_trialwise_2mm/sub-14/*.csv
glmPat  = fullfile(glmDir, sprintf('%s_%s_run-*_beta_index.csv', subID, taskName));
glmFiles = dir(glmPat);

for k = 1:numel(glmFiles)
    fpath = fullfile(glmFiles(k).folder, glmFiles(k).name);
    T = readtable(fpath); % CSV

    if ~ismember('odor_index', T.Properties.VariableNames)
        warning('[GLM] %s: no odor_index (skipped).', glmFiles(k).name);
        continue
    end

    % swap 1 <-> 2
    one = T.odor_index == 1;
    two = T.odor_index == 2;
    T.odor_index(one) = 2;
    T.odor_index(two) = 1;

    % backup + write
    backup_and_write(T, fpath, ',');
    fprintf('[GLM] %s | #1->2: %d, #2->1: %d\n', glmFiles(k).name, nnz(one), nnz(two));
end

%% --- Local helper (backup then write) -----------------------------------
function backup_and_write(T, fpath, delim)
    [p,b,e] = fileparts(fpath);
    backup = fullfile(p, [b '_orig' e]);
    if ~isfile(backup)
        copyfile(fpath, backup);
        fprintf('Backup created: %s\n', backup);
    else
        fprintf('Backup already exists: %s\n', backup);
    end
    writetable(T, fpath, 'FileType','text', 'Delimiter',delim);
end
