
% look at tSNR of PAID data
% focus on ROIs of interest
% OFC, LPFC
% do this on normalized MNI space in 2mm, before smoothing

clear; clc; close all

userHome = getenv('HOME');
HomeDir = fullfile(userHome,'Library/CloudStorage/Box-Box/NODEAP_data_analysis');

studydir = '/Volumes/X9Pro/NODEAP';
MRIdir = fullfile(studydir,'MRI');

SubIDlist = dir(fullfile(MRIdir, 'NODEAP*'));
SubIDlist = SubIDlist([SubIDlist.isdir]); % only keep directories
nSubIDlist = length(SubIDlist);

% read MRI count file
count_table = xlsread('/Volumes/X9Pro/NODEAP/MRI_func_count.xlsx');

rest_names = {'D0','S1D1','S1D2','S2D1','S2D2','S3D1','S3D2'};
n_rest_names = length(rest_names);

ROIdir = fullfile(studydir,'ConnectivityMasks');
maskpath = fullfile(HomeDir,'Scripts_matlab/FuncConn/mask');


%% write the session-wise tSNR map from PAID results

for subj = 1:nSubIDlist

SubID = SubIDlist(subj).name;
SubDir = fullfile(MRIdir,SubID);
niidir = fullfile(SubDir, 'nifti');
tSNPmapsdir = fullfile(niidir,'tSNPmaps');

for r = 1:n_rest_names % to loop over rest_names
    curr_rest = rest_names{r};
    fprintf('%s of %s\n',curr_rest,SubID)
    curr_func_dir = fullfile(niidir,'functional',curr_rest);
    
    % skip if either directory does not exist
    if count_table(subj,r)==0
        disp('Not enough data, skipping this iteration ...')
        continue;
    end
    
    fimage_name = fullfile(curr_func_dir, 's6w2fvol_4d.nii'); % use s6 & normalized data
    % compute voxel-wise tSNR
    hdr = spm_vol(sprintf('%s,1',fimage_name)); % get one header
    hdr.dt = [16,0];
    x = spm_read_vols(spm_vol(fimage_name));
    tSNR = mean(x,4)./std(x, [], 4);    % compute and save tSNR map
    
    hdr.fname = fullfile(tSNPmapsdir, sprintf('tSNR_PAID_%s.nii',curr_rest));
    spm_write_vol(hdr, tSNR);
    
end
end


%% load ROIs in 2mm MNI space

roiname = {'aOFC_seed_right','pOFC_seed_right','aOFC_target_right','pOFC_target_right'};
clear roiidx
for i = 1:length(roiname)
    roi_vol = spm_read_vols(spm_vol_nifti(fullfile(ROIdir,sprintf('%s.nii',roiname{i}))));
    roiidx{i} = find(roi_vol);
end

gm_file = fullfile(maskpath,'gm_0.1_2mm.nii');
roiidx{5} = find(spm_read_vols(spm_vol(gm_file)));
wm_file = fullfile(maskpath,'wm_0.9_2mm.nii');
roiidx{6} = find(spm_read_vols(spm_vol(wm_file)));
csf_file = fullfile(maskpath,'csf_0.9_2mm.nii');
roiidx{7} = find(spm_read_vols(spm_vol(csf_file)));

%% extract tSNR signal from ROIs

MeantSNRs = nan(n_rest_names,length(roiidx),nSubIDlist);

for subj = 1:nSubIDlist
    SubID = SubIDlist(subj).name; 
    SubDir = fullfile(MRIdir,SubID);
    niidir = fullfile(SubDir, 'nifti');
    tSNPmapsdir = fullfile(niidir,'tSNPmaps');

    for r = 1:n_rest_names % to loop over rest_names
        curr_rest = rest_names{r};
        
        fprintf('%s of %s\n',curr_rest,SubID)
    
        if count_table(subj,r)==0
            disp('Not enough data, skipping this iteration ...')
            continue;
        end
        
        for i = 1:length(roiidx) % loop over each ROI
            tSNRvalues = spm_read_vols(spm_vol(fullfile(tSNPmapsdir, sprintf('tSNR_PAID_%s.nii',curr_rest)))); 
            use_vals = tSNRvalues(roiidx{i});
            use_vals = use_vals(~isinf(use_vals));   % remove inf values
            MeantSNRs(r,i,subj) = nanmean(use_vals);
            
        end
    end
end

path = fullfile(HomeDir,'ProcessedData','tSNRs_PAID_subs_sessions.mat');
save(path, 'MeantSNRs'); 



