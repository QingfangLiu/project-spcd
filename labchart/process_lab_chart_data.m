
% this code processes labchart data to obtain processed breathing signals
% need the labchart data converted to .mat files already

clear; clc; close all;

addpath('/Users/liuq13/Applications/spm12/')

HomeDir = '/Users/liuq13/SPCD_data';
SubIDlist = dir(fullfile(HomeDir, 'Sub*'));
SubIDlist = SubIDlist([SubIDlist.isdir]); % only keep directories
nSubIDlist = length(SubIDlist);

SampleRate = 1000;

% chan1: task marker
% chan2 & 3: MFC display
% chan4: breathing traces from spirometer
% chan5: MRI triggers
trig_height = 5;  % trigger height from visualization

%%
for subj = 24 %1:nSubIDlist

    SubID = SubIDlist(subj).name;
    SubDir = fullfile(HomeDir,SubID);
    jobdir = fullfile(SubDir, 'jobs');
    SubLCDir = fullfile(SubDir,'labchart');

    tic

    fprintf('Working on %s\n',SubID)

for day = 1:2

    LC_Day_list = dir(fullfile(SubLCDir,sprintf('*MRI*Day%d.mat',day)));
    LC_mat = fullfile(LC_Day_list.folder,LC_Day_list.name);
    load(LC_mat)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % handle sub25-day1 differently by cutting the first 1.5e6 data
    if contains(SubID, '25') && day==1
        dat = dat(1.5e6+1:end, :);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % handle sub37-day2 differently by cutting the first 2e6 data
    if contains(SubID, '37') && day==2
        dat = dat(2e6+1:end, :);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % handle sub29-day2 differently by cutting the scanning run to be
    % discarded
    if contains(SubID, '29') && day==2
        start_loc = 9458780;
        end_loc = 10054800;
        dat(start_loc:end_loc, :) = [];
    end


    % handle sub11-day1 differently (labchart recording starts late)
    % handle sub7-day2 differently (labchart recording starts late)

    % handle sub7-day1 differently (redid odor scans in the end)
    % handle sub33-day1 differently (redid odor scans in the end)


    % handle sub35-day2 differently (missing MRI frames during run3 of odor
    % prediction task)

    
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

    img_name = fullfile(SubLCDir,sprintf('labchart_Day%d.bmp',day));
    saveas(gcf, img_name);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % number of volumes for Day 1 and Day 2 scan
    if day == 1
        num_vols = [320,320,ones(1,8)*244];
    else
        num_vols = [ones(1,3)*404,ones(1,6)*372];
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    allidx = find(dat(:,5) > trig_height);    % find all loc index with MRI trigger
    block_starts = [true; diff(allidx) > 1];  % denote the start of each frame (called block here)
    MRI_trig_idx = allidx(block_starts);      % find loc index of all frame starts
    fprintf('Number of MRI triggers detected: %d from Day %d\n',length(MRI_trig_idx),day)

    n_scan = length(num_vols);
    if sum(num_vols) ~= length(MRI_trig_idx)
        error('Wrong MRI trigger numbers detected. Stopping execution.');
    end

    MRI_trig_idx_segments = mat2cell(MRI_trig_idx', 1, num_vols); % MRI trig index into segments
    resp_data = dat(:,4);
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
        
        % keep flow rate, flow volume, and their squared values
        AllssR{r} = [ssR, ssR.^2, IntssR, IntssR.^2];

    end

    mat_name = sprintf('AllssR_Day%d.mat',day);
    save(fullfile(SubLCDir,mat_name), 'AllssR'); 

    toc

end
end