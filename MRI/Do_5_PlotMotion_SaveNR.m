
% This code
% 0. arrange the breathing data according to expected run order
% 1. plot realignment motion parameters across volumes, for each run
% and each subject
% 2. do within-volumn calculation 
% for multi-echo data, motion estimated from e1
% 3. saves everything as nuisance regressor

% before running this 
% need to have processed the labchart data to obtain the breathing signals

clc; clear; close all;

%% setup

addpath('/Users/liuq13/Applications/spm12/')
addpath('/Users/liuq13/xjview10/xjview/')

HomeDir = '/Users/liuq13/SPCD_data';
SubIDlist = dir(fullfile(HomeDir, 'Sub*'));
SubIDlist = SubIDlist([SubIDlist.isdir]); % only keep directories
nSubIDlist = length(SubIDlist);

pattern_list = {'place', 'character', 'odor_scan', 'odorpred', 'premeal', 'postmeal'}; 

%%
for subj = 1:nSubIDlist

    SubID = SubIDlist(subj).name;
    SubDir = fullfile(HomeDir,SubID);
    SubLCDir = fullfile(SubDir,'labchart');

    Day1mat = fullfile(SubLCDir,'AllssR_Day1.mat');
    Day2mat = fullfile(SubLCDir,'AllssR_Day2.mat');

    if ~(exist(Day1mat,'file') && exist(Day2mat,'file'))
        continue;  % if this subj does not contain the two .mat files, proceed to next subj
    end

    % where to save nuisance regressors
    NROutdir = fullfile(SubDir,'NRegressor');
    if ~exist(NROutdir,'dir')
        mkdir(NROutdir)
    end

    if exist(fullfile(NROutdir,'BadVolumes.mat'),'file')
        continue;  % if this subj has been processed, proceed to next subj
    end

    disp(['Processing: ', SubID]);
    
    mpfilename = dir(fullfile(SubDir, 'nii', 'func', '*', 'rp*.txt')); 
    nruns = length(mpfilename);
    
    % initiate some variables for curr subject
    slicediff = cell(1,nruns);
    slicevar = cell(1,nruns);
    adj_slicediff = cell(1,nruns);
    adj_slicevar = cell(1,nruns);
    NR = cell(1,nruns);
    badvol = cell(1,nruns);      % bad volumes per session
    ctr_badvol = zeros(nruns,1); % count # of bad volumes
    ctr_NR = zeros(nruns,1);     % count # of nuisance regressors
    
    nscans = nan(nruns,1);       % # scans per run
    run_string = cell(nruns,1);  % unique run string saved as cell
    
    
  
    % load preprocessed breathing data
    % then rearrange to expected run order here
    % character run 1,2,3,4: from Day 1 run 3,4,5,6
    % odor_scan run 1,2: from Day 1 run 1,2
    % odorpred run 1,2,3: from Day 2 run 1,2,3
    % place run 1,2,3,4: from Day 1 run 7,8,9,10
    % postmeal run 1,2,3: from Day 2 run 7,8,9
    % premeal run 1,2,3: from Day 2 run 4,5,6

    load(Day1mat);
    AllssR_Day1 = AllssR;
    load(Day2mat);
    AllssR_Day2 = AllssR;

    AllssR = cell(1,nruns);
    AllssR(1:4) = AllssR_Day1(3:6);
    AllssR(5:6) = AllssR_Day1(1:2);
    AllssR(7:9) = AllssR_Day2(1:3);
    AllssR(10:13) = AllssR_Day1(7:10);
    AllssR(14:16) = AllssR_Day2(7:9);
    AllssR(17:19) = AllssR_Day2(4:6);

for r = 1:nruns
   
    input_str = mpfilename(r).name;
    pattern = strjoin(pattern_list, '|');
    match = regexp(input_str, ['(?:' pattern ')(_run_\d+)'], 'tokens', 'once');
    if ~isempty(match)
        match = strcat(regexp(input_str, pattern, 'match', 'once'), match{1});
    end

    run_string{r,1} = match;
    fprintf('%s of %s\n',match,SubID)
    
    % find the motion parameters
    mp = load(fullfile(mpfilename(r).folder, mpfilename(r).name));          % mp motion parameters
    nscans(r) = size(mp,1);     

    % add diff, squared mp, and squared diff
    NR{r} = [mp, [zeros(1,6); diff(mp)], mp.^2, [zeros(1,6); diff(mp).^2]];

    figure;
    subplot(2,4,1); plot(NR{r}(:,1:3)); xlabel('scans'); ylabel('mm'); 
    subplot(2,4,2); plot(NR{r}(:,4:6)); xlabel('scans'); ylabel('radian');
    subplot(2,4,3); plot(NR{r}(:,7:9)); xlabel('scans'); ylabel('diff(mm)'); 
    subplot(2,4,4); plot(NR{r}(:,10:12)); xlabel('scans'); ylabel('diff(radian)');
    subplot(2,4,5); plot(NR{r}(:,13:15)); xlabel('scans'); ylabel('mm^2');
    subplot(2,4,6); plot(NR{r}(:,16:18)); xlabel('scans'); ylabel('radian^2');
    subplot(2,4,7); plot(NR{r}(:,19:21)); xlabel('scans'); ylabel('diff(mm)^2'); 
    subplot(2,4,8); plot(NR{r}(:,22:24)); xlabel('scans'); ylabel('diff(radian)^2');
    set(gcf, 'PaperPosition', [2 1 18 8]);

    % save the motion parameter plot
    tmpfigname = fullfile(NROutdir,sprintf('MotionParameters_%s_%s.bmp',SubID,match));
    saveas(gcf,tmpfigname)

    % add four breathing traces to nuisance regressor
    NR{r} = [AllssR{r}, NR{r}];
       
%% analyze slice differences and variances
    % this is done using e1 before any processing (as we did realignment using e1)
    idx1 = 1:2:60;   % odd slices
    idx2 = 2:2:60;   % even slices

    funcfilenames = dir(fullfile(mpfilename(r).folder,sprintf('Scan*%s*_e1.nii',match))); 
    y = spm_vol(fullfile(mpfilename(r).folder,funcfilenames(1).name)); 
    
    for vol = 1:nscans(r)
        v = spm_read_vols(y(vol));   %dimensions of v show that there are 60 slices
        odd_slice = nanmean(nanmean(nanmean(v(:,:,idx1),3)));
        even_slice = nanmean(nanmean(nanmean(v(:,:,idx2),3)));
        slicediff{r}(vol) = odd_slice - even_slice; % diff b/t odd and even slices
        slicevar{r}(vol) = var(nanmean(squeeze(nanmean(v)))); % var across all 60 slices
    end

end % end of the run

close all

all_slicediff = [slicediff{:}]; % aggregate all slice differences across sessions and runs
all_slicevar = [slicevar{:}]; % aggregate all slice variances across sessions and runs
aztmp = [max(abs(zscore(all_slicediff))),max(abs(zscore(all_slicevar)))];
crit = [4,4];        % criterion for slicediff and slicevar, eyeball it looking at the peaks

figure
for r = 1:nruns
           
        adj_slicediff{r} = abs((slicediff{r} - mean(all_slicediff))./std(all_slicediff));
        adj_slicevar{r} = abs((slicevar{r} - mean(all_slicevar))./std(all_slicevar));

        sh = subplot(nruns, 2, r*2-1);    
        plot(adj_slicediff{r})
        set(sh, 'YLim', [0,max(aztmp)+1]);
        title(sprintf('slice diff, %s',run_string{r}),'Interpreter', 'none');
        hold on
        plot(find(adj_slicediff{r}> crit(1)), adj_slicediff{r}(adj_slicediff{r}> crit(1)), '*r');
        
        sh = subplot(nruns, 2, r*2);
        plot(adj_slicevar{r})
        set(sh, 'YLim', [0,max(aztmp)+1]);
        title(sprintf('slice var, %s',run_string{r}),'Interpreter', 'none');
        hold on
        plot(find(adj_slicevar{r}> crit(2)), adj_slicevar{r}(adj_slicevar{r}> crit(2)), '*r');
        
        % include regressors for bad volumes
        tmpbadvol = unique([find(adj_slicediff{r}> crit(1)), find(adj_slicevar{r}> crit(2))]);
        badvol{r} = tmpbadvol;
        ctr_badvol(r) = length(tmpbadvol);         % count bad volumes
        bv = zeros(nscans(r),ctr_badvol(r));
        for i = 1:ctr_badvol(r)
            bv(tmpbadvol(i),i) = 1;
        end
            
        % add slicediff and diff of slicediff and bad volume regressors
        NR{r} = [NR{r},adj_slicediff{r}', adj_slicevar{r}',...
            [0; diff(adj_slicediff{r})'], [0; diff(adj_slicevar{r})'],...
            [0; diff(adj_slicediff{r})'].^2, [0; diff(adj_slicevar{r})'].^2,...
            bv];   
        ctr_NR(r) = size(NR{r},2);
        
        % save z-scored nuisance regressors as txt
        NRfilename = fullfile(NROutdir,sprintf('nuisance_regressors_%s.txt', run_string{r}));
        dlmwrite(NRfilename, NR{r}, 'delimiter', ' ', 'precision', '%.24f');   % no z-scored

        NRfilename = fullfile(NROutdir,sprintf('nuisance_regressors_%s_zscored.txt', run_string{r}));
        dlmwrite(NRfilename, zscore(NR{r}), 'delimiter', ' ', 'precision', '%.24f');   % z-scored
    
end 

        % Save Slice diff and var figures 
        set(gcf, 'PaperPosition', [2 1 12 32]);
        saveas(gcf,fullfile(NROutdir,'SliceDiffVar.bmp'))
        
        % save variables for current subject: badvol, ctr_badvol, ctr_NR
        save(fullfile(NROutdir,'BadVolumes.mat'), 'badvol', 'ctr_badvol', 'ctr_NR');  
        
        close all

end % subject

