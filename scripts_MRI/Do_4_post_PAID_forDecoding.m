

% This code continues processing rt-fMRI data on PAID data.
% for decoding purpose
% smooth data with 2mm FWHM

spm('Defaults','fMRI')
global defaults
defaults.stats.maxmem = 2 ^ 34; % based on RAM = 32GB
defaults.stats.resmem = true; % allow temp files to be kept in memory

%%
func_4d = {};    
n = dir(fullfile(SubDir,'func_PAID', '*', 'w2*.nii')); 
nruns = length(n);
for r = 1:nruns
    func_4d{r,1} = fullfile(n(r).folder,n(r).name);
end

if isempty(dir(fullfile(SubDir,'func_PAID','*','s2w2*.nii')))
    matlabbatch = [];
    
    % Smooth EPI to 2mm 
    matlabbatch{1}.spm.spatial.smooth.data = func_4d;
    matlabbatch{1}.spm.spatial.smooth.fwhm = [2 2 2];
    matlabbatch{1}.spm.spatial.smooth.dtype = 0;
    matlabbatch{1}.spm.spatial.smooth.im = 0;
    matlabbatch{1}.spm.spatial.smooth.prefix = 's2';

    % Run preprocessing
    job_name = fullfile(jobdir, 'Post_PAID_Processing2.mat');
    save(job_name, 'matlabbatch');
    spm_jobman('run', matlabbatch)
    clear matlabbatch

end
