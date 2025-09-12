
% this code processes labchart data to obtain processed breathing signals
% use the .mat labchart data 

% what has been addressed:
    % handle sub7-day1 differently (redid odor scans in the end)
    % handle sub33-day1 differently (redid odor scans in the end)

% Special handling for sub-11-day1 and sub7-day2:
% LabChart recording starts late, missing part of run 1.
% We keep the available data and pad the beginning with NaNs 
% to maintain expected length (320 volumes) for nuisance regression.

% chan1: task marker
% chan2 & 3: MFC display
% chan4: breathing traces from spirometer
% chan5: MRI triggers


clear; clc; close all;

addpath('/Users/liuq13/Applications/spm12/')

HomeDir = '/Users/liuq13/SPCD_data/labchart_processed';
SubIDlist = dir(fullfile(HomeDir, 'sub-*'));
SubIDlist = SubIDlist([SubIDlist.isdir]); % only keep directories
nSubIDlist = length(SubIDlist);

SampleRate = 1000;
trig_height = 5;  % trigger height from visualization

%%
for subj = 29 %1:nSubIDlist

    SubID = SubIDlist(subj).name;    
    SubLCDir = fullfile(HomeDir,SubID);

    tic

    fprintf('Working on %s\n',SubID)

for day = 1:2

    LC_Day_list = dir(fullfile(SubLCDir,sprintf('day%d',day),sprintf('*MRI*Day%d.mat',day)));
    LC_mat = fullfile(LC_Day_list.folder,LC_Day_list.name);
    load(LC_mat)

    % ==== Pre-cleaning for specific known issues ====
    if strcmp(SubID, 'sub-25') && day == 1
        dat = dat(1.5e6+1:end, :);
    elseif strcmp(SubID, 'sub-36') && day == 1
        dat(1.8e6:3e6, :) = [];
    elseif strcmp(SubID, 'sub-37') && day == 2
        dat = dat(2e6+1:end, :);
    elseif strcmp(SubID, 'sub-29') && day == 2
        dat(9458780:10054800, :) = [];
    elseif strcmp(SubID, 'sub-38') && day == 2
        dat(2.8e6:3e6, :) = [];
    end

    % handle sub35-day2 differently (missing MRI frames during run3 of odor
    % prediction task)
    is_partial_third_run_day2 = strcmp(SubID, 'sub-35') && day == 2;

    % handle sub36-day2 differently (two parts of scan for pre-meal run3)
    is_split_run_day2 = strcmp(SubID, 'sub-36') && day == 2;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % take a quick look at the labchart data
    % Day 1 contains: 2 runs of odor scan, 4 runs of char scan, 4 runs of
    % place scan
    % Day 2 contains: 3 runs of odor prediction, 3 runs of pre-meal test, 3
    % runs of post-meal test

    figure('Visible', 'off', 'Position', [100, 100, 2400, 1600]); % plot without showing
    for i = 1:5
        subplot(5,1,i)
        plot(dat(:,i))
    end

    img_name = fullfile(SubLCDir,sprintf('day%d',day),sprintf('labchart_Day%d.bmp',day));
    saveas(gcf, img_name);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % number of volumes for Day 1 and Day 2 scan
    if day == 1
        default_vols = [repmat(320, 1, 2), repmat(244, 1, 8)];  % 2 odor + 4 character + 4 place
    else
        default_vols = [repmat(404, 1, 3), repmat(372, 1, 6)];  % Day 2 structure
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    allidx = find(dat(:,5) > trig_height);    % find all loc index with MRI trigger
    block_starts = [true; diff(allidx) > 1];  % denote the start of each frame (called block here)
    MRI_trig_idx = allidx(block_starts);      % find loc index of all frame starts
    fprintf('Number of MRI triggers detected: %d from Day %d\n',length(MRI_trig_idx),day)

    % === Special case flag
    is_extra_odor_runs_day1 = ismember(SubID, {'sub-07', 'sub-33'}) && day == 1;
    is_partial_first_run_day1 = strcmp(SubID, 'sub-11') && day == 1;
    is_partial_first_run_day2 = strcmp(SubID, 'sub-07') && day == 2;

    % === Segment trigger indices
    if is_extra_odor_runs_day1
        full_vols = [repmat(320, 1, 2), repmat(244, 1, 8), repmat(320, 1, 2)];
        if length(MRI_trig_idx) ~= sum(full_vols)
            error('Unexpected number of triggers for Day 1.');
        end
    
        full_segments = mat2cell(MRI_trig_idx', 1, full_vols);
        % Use final 2 runs (as odor), plus middle 8 (character/place)
        MRI_trig_idx_segments = [full_segments(11:12), full_segments(3:10)];
        num_vols = [320, 320, repmat(244, 1, 8)];  % Correct output format
        fprintf('Day 1: using final 2 odor runs and middle 8 valid runs.\n');
    
    elseif is_partial_first_run_day1 || is_partial_first_run_day2
        expected_vols = sum(default_vols);         % 2592
        actual_vols = length(MRI_trig_idx);        % e.g. 2572
        missing = expected_vols - actual_vols;     % e.g. 20

        fprintf('%s Day %d: Partial run 1 (%d triggers missing), will pad with zeros.\n', SubID, day, missing);
        partial_run1_len = default_vols(1) - missing;
        seg_lengths = [partial_run1_len, default_vols(2:end)];
        cut_points = [0, cumsum(seg_lengths)];
        MRI_trig_idx_segments = cell(1, length(default_vols));
        for r = 1:length(default_vols)
            MRI_trig_idx_segments{r} = MRI_trig_idx(cut_points(r)+1 : cut_points(r+1));
        end
        num_vols = default_vols;
        
    elseif is_partial_third_run_day2
        
        n_missing = sum(default_vols) - length(MRI_trig_idx);
        fprintf('Detected %d missing volumes — adjusting last odor run.\n', n_missing);
        adjusted_vols = default_vols;
        adjusted_vols(3) = adjusted_vols(3) - n_missing;  % Only modify 3rd odor run
        MRI_trig_idx_segments = mat2cell(MRI_trig_idx', 1, adjusted_vols);
        num_vols = adjusted_vols;
        
    elseif is_split_run_day2

        fprintf('⚠️ Excluding interrupted pre-meal run 3 (Run 6)\n');

        % Volumes you want to keep
        vols_keep_first = 404 + 404 + 404 + 372 + 372;  % = 1956
        vols_keep_last = 3 * 372;                      % = 1116
        
        % Use MRI_trig_idx directly
        MRI_trig_idx_clean = [
        MRI_trig_idx(1:vols_keep_first);
        MRI_trig_idx(end - vols_keep_last + 1:end)];

        % New volume structure (without run 6)
        default_vols = [404, 404, 404, 372, 372, 372, 372, 372];

        % Assign cleaned result
        MRI_trig_idx = MRI_trig_idx_clean;

        MRI_trig_idx_segments = mat2cell(MRI_trig_idx', 1, default_vols);
        num_vols = default_vols;
        
    
    else % normally without interruption

        if sum(default_vols) ~= length(MRI_trig_idx)
            error('Mismatch in expected vs actual MRI triggers. Stopping.');
        end
        MRI_trig_idx_segments = mat2cell(MRI_trig_idx', 1, default_vols);
        num_vols = default_vols;
    end
    
    resp_data = dat(:,4);

    n_scan = length(num_vols);
    AllssR = cell(1,n_scan);

    for r = 1:n_scan

        start_end_idx = MRI_trig_idx_segments{r}([1,end]);
        start_end_idx(2) = start_end_idx(2) + 2;          % add two samples for convenience next
        R = resp_data(start_end_idx(1):start_end_idx(2)); % truncate the breathing data  
        R = smooth(R,250);                                % smooth breathing data
        R = R(1:100:end);                                 % downsample breathing data
        sr = SampleRate/100;
    
        % high pass filter
        K.RT = 1/sr;
        K.row = ones(length(R),1);
        K.HParam = 50;
        R = spm_filter(K,R);
    
        R = zscore(R);       % Zscore R before integration - important!
        IntR = cumtrapz(R);  % integration to get breathing volume
        IntR = zscore(IntR); % Zscore IntR
    
        % down sample to match scan resolution 1500ms
        ssR = R(1:15:end);
        IntssR = IntR(1:15:end);
        this_run = [ssR, ssR.^2, IntssR, IntssR.^2];
        
        % Handle any subject with partial first run (e.g., sub-11 Day 1, sub-07 Day 2)
        if r == 1 && ...
        ((strcmp(SubID, 'sub-11') && day == 1) || (strcmp(SubID, 'sub-07') && day == 2))
            expected_len = num_vols(1);
            actual_len = size(this_run, 1);
            missing = expected_len - actual_len;
            fprintf('%s Day %d Run 1: Detected %d missing volumes. Padding with Zeros at start.\n', SubID, day, missing);
            this_run = [zeros(missing, 4); this_run];
        end

        AllssR{r} = this_run;

    end

    if is_split_run_day2
        % Insert empty cell at position 6
        AllssR = [AllssR(1:5), {[]}, AllssR(6:end)];
    end

    mat_name = sprintf('AllssR_Day%d.mat',day);
    save(fullfile(SubLCDir,sprintf('day%d',day),mat_name), 'AllssR'); 

    toc

end
end