
addpath('/Users/liuq13/Applications/spm12/')
addpath('/Users/liuq13/xjview10/xjview/')

HomeDir = '/Users/liuq13/SPCD_data/nii';
SubIDlist = dir(fullfile(HomeDir, 'sub-*'));
SubIDlist = SubIDlist([SubIDlist.isdir]); % only keep directories
nSubIDlist = length(SubIDlist);

TEs = [12.60, 29.16, 45.72, 62.28];
nechos = 4;

dcm2niipath = '/Applications/MRIcroGL.app/Contents/Resources/dcm2niix';
size_threshold = 2 * 1024 * 1024; % Set size threshold (2 MB)
pattern_list = {'place', 'character', 'odor_scan', 'odorpred', 'premeal', 'postmeal'}; 

%%

for subj = 5

    SubID = SubIDlist(subj).name;
    disp(['Processing: ', SubID]);
    SubDir = fullfile(HomeDir,SubID);
    jobdir = fullfile(SubDir, 'jobs');
    if ~exist(jobdir,'dir')
        mkdir(jobdir)
    end

    % a series of code
    %disp('Code: Do_1_Convert_realign');
    %Do_1_Convert_realign
    %disp('Code: Do_2_Reslice');
    %Do_2_Reslice
    %disp('Code: Do_3_cal_mean');
    %Do_3_cal_mean
    disp('Code: Do_4_post_PAID');
    Do_4_post_PAID
    disp('Code: Do_4_post_PAID_forDecoding');
    Do_4_post_PAID_forDecoding

    %Do_1_half_recopy_raw_nifti

end







