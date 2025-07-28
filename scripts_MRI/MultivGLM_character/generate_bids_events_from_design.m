
% ================================================================
% This script generates BIDS-compatible events.tsv files for
% the character image scan task using subject-specific Excel files.
%
% – Iterates over subjects and multiple runs
% – Determines whether to use even/odd design based on pairing type
% – Loads timing info from 'IndepScanConds_<even/odd>_idx*.xlsx'
% – Computes image onset times using fixed cue/image durations and ITIs
% – Keeps only task trials (where correct == 'null'); excludes filler trials
% – Saves image onset, duration (2s), and trial_type (img_<cue_idx>) to .tsv
% – Output path: /nii/sub-XX/func_PAID/character/
% ================================================================

clear; clc;

% ---------- USER CONFIG ----------
HomeDir = '/Users/liuq13/SPCD_data';
niiDir = fullfile(HomeDir, 'nii');
GLMDir = fullfile(HomeDir, 'GLM');

% Load subject info for pairing assignment
subject_table = readtable(fullfile(HomeDir, 'task_info', 'subject_info.csv'), 'VariableNamingRule', 'preserve');

% Get list of subject folders (those starting with 'sub-')
allEntries = dir(fullfile(niiDir, 'sub-*'));
subjectFolders = allEntries([allEntries.isdir]);  % keep only directories

runs = 1:4;  % 4 char image runs

% Cue timing
blank_dur = 10;
cross_dur = 2;
image_dur = 2;

%% ---------- LOOP ----------
for subj = 1:length(subjectFolders)
    SubID = subjectFolders(subj).name;
    fprintf('\n\n=== Processing: %s ===\n', SubID);

    % Determine design file version based on pairing type
    pair_row = strcmp(subject_table.subject, SubID);
    pair_type = subject_table.pairing{pair_row};  % 'p' or 'c'
    version = "even";
    if strcmp(pair_type, 'p')
        version = "odd";
    end

    for r = runs
        % File paths
        design_file = sprintf('IndepScanConds_%s_idx%d.xlsx', version, r);
        input_path = fullfile(HomeDir, 'beh_raw', SubID, design_file);
        output_dir = fullfile(HomeDir, "nii", SubID, 'func_PAID', 'character');
        output_file = fullfile(output_dir, ...
            sprintf('%s_task-character_run-%d_events.tsv', SubID, r));
        
        % Read Excel design
        try
            design = readtable(input_path);
        catch
            fprintf('❌ Failed to read: %s\n', input_path);
            continue;
        end

        
        ntrials = height(design);

        % Compute sniff cue onsets
        img_onsets = nan(ntrials, 1);
        img_onsets(1) = blank_dur + cross_dur;
        for t = 2:ntrials
            img_onsets(t) = img_onsets(t-1) + cross_dur + image_dur + design.ITIDur(t-1);
        end

        design.img_onset = img_onsets;

        % Keep only task trials
        is_task = strcmp(design.correct, 'null');
        design = design(is_task, :);

        % Create table
        events = table();
        events.onset = design.img_onset;
        events.duration = repmat(image_dur, height(design), 1);
        events.trial_type = strcat("img_", string(design.cue_idx));

        % Save as BIDS .tsv
        writetable(events, output_file, 'FileType', 'text', 'Delimiter', '\t');
        fprintf('✅ Saved: %s\n', output_file);
        
    end
end
