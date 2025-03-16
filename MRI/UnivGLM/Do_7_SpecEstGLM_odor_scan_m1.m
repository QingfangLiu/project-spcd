

% specify a decoding GLM from the odor scan runs

clear; clc;

TR = 1.5;

subj = 1;
SubID = SubIDlist(subj).name;
disp(['Processing: ', SubID]);
SubDir = fullfile(HomeDir,SubID);
jobdir = fullfile(SubDir, 'jobs');

modelname = 'odor_scan_m1';
modeldir = fullfile(SubDir,'Univariate',modelname);
      
clear matlabbatch
matlabbatch{1}.spm.stats.fmri_spec.dir = {modeldir};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

nruns = 2; % two runs of odor scan
for r = 1:nruns
    
    scandir = fullfile(SubDir,'nii','func_PAID','odor_scan');
    Scan_4D = dir(fullfile(scandir,sprintf('s6w2*run_%d*.nii',r))); % using the 6mm smoothed images
    im_name = fullfile(scandir,Scan_4D.name);
    ims = spm_select('expand', im_name);  % get 3D image names from 4D
    ims = cellstr(ims);                   % convert char to cell

    onsfile = fullfile(modeldir,sprintf('Run%d_Onsets.mat',r));
    regfile = fullfile(SubDir,'NRegressor',sprintf('nuisance_regressors_odor_scan_run_%d_zscored.txt',r)); % nuisance regressor

    matlabbatch{1}.spm.stats.fmri_spec.sess(r).scans = ims;
    matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(r).multi = {onsfile};
    matlabbatch{1}.spm.stats.fmri_spec.sess(r).regress = struct('name', {}, 'val', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(r).multi_reg = {regfile};
    matlabbatch{1}.spm.stats.fmri_spec.sess(r).hpf = 128;

end

matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.1;
matlabbatch{1}.spm.stats.fmri_spec.mask = {''}; 
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(modeldir,'SPM.mat')};
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

fname = fullfile(jobdir, sprintf('SpecEstGLM_Univariate_%s_%s.mat',modelname,SubID));
save(fname, 'matlabbatch');

spm fmri
spm_jobman('run', matlabbatch);   