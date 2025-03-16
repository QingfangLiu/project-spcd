
% This script is used to find the time onsets of task events aligned with
% the scan timing.
% do this for the odor scan task

clear; clc;

%%

subj = 1;
SubID = SubIDlist(subj).name;
disp(['Processing: ', SubID]);
SubDir = fullfile(HomeDir,SubID);

modelname = 'odor_scan_m1';
modeldir = fullfile(SubDir,'univariate',modelname);
if ~exist(modeldir,'dir')
    mkdir(modeldir)
else
end

% calculate time onsets
% 10s blank after first volume
% each trial: 2s white cross, 2s blue cross, ITI
% 30trials/run
blank = 10;
white_cross = 2;
blue_cross = 2;

%% loop over runs
nruns = 2;
for r = 1:nruns

    fprintf('Run%d\n',r)

    % load task design xls
    task_design_file = sprintf('IndepScanOdorConds%d.xlsx',r);
    task_design = readtable(fullfile(SubDir,SubID,task_design_file));
    odor_num = task_design.odor_num;
    ITIDur = task_design.ITIDur;
    ntrials = size(task_design,1);
    
    odor_0_idx = odor_num == 0;
    odor_1_idx = odor_num == 1;
    odor_2_idx = odor_num == 2;
    
    sniff_onsets = nan(1,ntrials);    
    sniff_onsets(1) = blank + white_cross;
    
    for t = 2:ntrials
        sniff_onsets(t) = sniff_onsets(t-1) + blue_cross + ITIDur(t-1) + white_cross;
    end
    
    odor_0_onsets = sniff_onsets(odor_0_idx);
    odor_1_onsets = sniff_onsets(odor_1_idx);
    odor_2_onsets = sniff_onsets(odor_2_idx);

    names{1} = 'odor0';
    names{2} = 'odor1';
    names{3} = 'odor2';
        
    onsets{1} = odor_0_onsets;
    onsets{2} = odor_1_onsets; 
    onsets{3} = odor_2_onsets;
    
    num_onsets = length(onsets);
    durations = cell(1,num_onsets);
    for i = 1:num_onsets
        durations{i} = zeros(size(onsets{i}));
    end

    matfile = fullfile(modeldir,sprintf('Run%d_Onsets.mat',r));
    save(matfile,'names','onsets','durations');

end 
