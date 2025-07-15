

modelname = 'odor_scan';
modeldir = fullfile(GLMDir, modelname, SubID);
      
clear matlabbatch
matlabbatch{1}.spm.stats.fmri_spec.dir = {modeldir};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

for r = 1:nruns
    
    scandir = fullfile(niiDir, SubID, 'func_PAID', 'odor_scan');

    % First try to find the .nii.gz file
    gz_files = dir(fullfile(scandir, sprintf('s6w2*run_%d*.nii.gz', r)));

    if ~isempty(gz_files)
        gz_file = fullfile(scandir, gz_files(1).name);
        [~, name_noext] = fileparts(gz_file);  % removes .gz
        nii_file = fullfile(scandir, name_noext);
    
        if ~exist(nii_file, 'file')
            fprintf('🌀 Unzipping %s\n', gz_file);
            gunzip(gz_file, scandir);
        end

    elseif isempty(gz_files)
        % Try to find .nii file directly
        nii_files = dir(fullfile(scandir, sprintf('s6w2*run_%d*.nii', r)));
    
        if isempty(nii_files)
            error('❌ No .nii or .nii.gz file found for run %d in %s', r, scandir);
        end
    
        nii_file = fullfile(scandir, nii_files(1).name);
    end

    ims = spm_select('expand', nii_file);  % get 3D image names from 4D
    ims = cellstr(ims);                    % convert char to cell

    onsfile = fullfile(modeldir,sprintf('Run%d_Onsets.mat',r));
    regfile = fullfile(HomeDir,'NRegressor',SubID,sprintf('nuisance_regressors_odor_scan_run_%d_zscored.txt',r)); % nuisance regressor

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

