

% This script computes onsets for the odor scan task aligned to scan timing.
% Each trial: 2s white cross → 2s sniff cue (blue cross, with odor) → ITI
% Odor starts at 2s after trial start. There is a 10s blank before trial 1.

% 30trials/run, intermixed with clean air, odor1 and odor2


modelname = 'odor_scan';
modeldir = fullfile(GLMDir, modelname, SubID);
if ~exist(modeldir,'dir')
    mkdir(modeldir)
else
end

% Constants
blank_dur = 10;
white_cross_dur = 2;
blue_cross_dur = 2;
nruns = 2;

for r = 1:nruns

    % load task design
    task_design_file = sprintf('IndepScanOdorConds%d.xlsx',r);
    task_design = readtable(fullfile(HomeDir,'beh_raw',SubID,task_design_file));
    
    odor_num = task_design.odor_num; % 0 = clean, 1 = odor1, 2 = odor2
    ITIDur = task_design.ITIDur;
    ntrials = size(task_design,1);
    
    % Preallocate sniff cue onsets
    sniff_onsets = nan(1, ntrials);
    sniff_onsets(1) = blank_dur + white_cross_dur;

    for t = 2:ntrials
        sniff_onsets(t) = sniff_onsets(t-1) + white_cross_dur + blue_cross_dur + ITIDur(t-1);
    end

    % Split by condition
    onsets = {
        sniff_onsets(odor_num == 0), ...
        sniff_onsets(odor_num == 1), ...
        sniff_onsets(odor_num == 2)
    };
    
    names = {'odor0', 'odor1', 'odor2'};
    durations = cellfun(@(x) repmat(blue_cross_dur, size(x)), onsets, 'UniformOutput', false);
    
    matfile = fullfile(modeldir,sprintf('Run%d_Onsets.mat',r));
    save(matfile,'names','onsets','durations');

end 
