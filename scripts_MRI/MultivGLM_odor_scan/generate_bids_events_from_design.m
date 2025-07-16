
clear; clc;

% ---------- USER CONFIG ----------
HomeDir = '/Users/liuq13/SPCD_data';
niiDir = fullfile(HomeDir, 'nii');
GLMDir = fullfile(HomeDir, 'GLM');

% Get list of subject folders (those starting with 'sub-')
allEntries = dir(fullfile(niiDir, 'sub-*'));
subjectFolders = allEntries([allEntries.isdir]);  % keep only directories

runs = 1:2;  % Change based on how many runs each subject has

stim_duration = 2;  % Odor delivery duration (s)

% Cue timing
blank_dur       = 10;
white_cross_dur = 2;
blue_cross_dur  = 2;

% ---------- LOOP ----------
for subj = 1:length(subjectFolders)
    SubID = subjectFolders(subj).name;
    fprintf('\n\n=== Processing: %s ===\n', SubID);
    
    for r = runs
        % File paths
        design_file = sprintf('IndepScanOdorConds%d.xlsx', r);
        input_path = fullfile(HomeDir, 'beh_raw', SubID, design_file);
        output_dir = fullfile(HomeDir, "nii", SubID, 'func_PAID', 'odor_scan');
        output_file = fullfile(output_dir, ...
            sprintf('%s_task-odor_run-%d_events.tsv', SubID, r));
        
        % Skip if design file doesn't exist
        if ~isfile(input_path)
            fprintf('⚠️  Missing design file: %s\n', input_path);
            continue;
        end
        
        % Read Excel design
        try
            design = readtable(input_path);
        catch
            fprintf('❌ Failed to read: %s\n', input_path);
            continue;
        end

        ntrials = height(design);
        
        % Compute sniff cue onsets
        sniff_onsets = nan(ntrials, 1);
        sniff_onsets(1) = blank_dur + white_cross_dur;
        for t = 2:ntrials
            sniff_onsets(t) = sniff_onsets(t-1) + white_cross_dur + blue_cross_dur + design.ITIDur(t-1);
        end
        design.onset = sniff_onsets;
        
        % Set duration
        design.duration = repmat(stim_duration, ntrials, 1);
        
        % Map trial_type
        odor_map = containers.Map([0 1 2], {'clean_air', 'odor_1', 'odor_2'});
        design.trial_type = cell(ntrials, 1);
        for t = 1:ntrials
            odor_val = design.odor_num(t);
            if isKey(odor_map, odor_val)
                design.trial_type{t} = odor_map(odor_val);
            else
                design.trial_type{t} = 'unknown';
            end
        end

        % Save as BIDS .tsv
        events = table(design.onset, design.duration, design.trial_type, ...
            'VariableNames', {'onset','duration','trial_type'});
        writetable(events, output_file, 'FileType', 'text', 'Delimiter', '\t');
        
        fprintf('✅ Saved: %s\n', output_file);
    end
end
