

% This code continues processing rt-fMRI data on PAID data.
% steps include: coregistration, normalization and smoothing

spm('Defaults','fMRI')
global defaults
defaults.stats.maxmem = 2 ^ 34; % based on RAM = 32GB
defaults.stats.resmem = true; % allow temp files to be kept in memory

%%

% get anatomical image
aname = dir(fullfile(SubDir,'anat','*MPRAGE*.nii'));
afile = fullfile(SubDir,'anat',aname(1).name);
% mean func vol across sessions
func_mean = fullfile(SubDir,'func_PAID', 'mean_fvol.nii');

if isempty(dir(fullfile(SubDir,'anat','w*.nii')))
    % Normalize T1 to generate warping parameters
    matlabbatch = [];
    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol = {afile};
    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {afile};
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.0001;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm =  {fullfile(spm('dir'),'tpm','TPM.nii')};
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.affreg = 'mni';
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.samp = 3;
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.bb = [-78 -112 -70
        78 76 85];
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.vox = [1 1 1];
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.interp = 4;

    % Run preprocessing
    job_name = fullfile(jobdir, 'Normalization.mat');
    save(job_name, 'matlabbatch');
    spm_jobman('run', matlabbatch)
    clear matlabbatch
end

    func_4d = {};    
    n = dir(fullfile(SubDir, 'func_PAID', '*', '*.nii')); 
    nruns = length(n);
    for r = 1:nruns
        func_4d{r,1} = fullfile(n(r).folder,n(r).name);
    end
    
    if isempty(dir(fullfile(SubDir,'func_PAID','*','s6w2*.nii')))
        matlabbatch = [];
        % Coregister mean realigned EPI to T1, and bring along the realigned EPIs 
        matlabbatch{1}.spm.spatial.coreg.estimate.ref = {afile};
        matlabbatch{1}.spm.spatial.coreg.estimate.source = {func_mean};
        matlabbatch{1}.spm.spatial.coreg.estimate.other = func_4d;
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];

        % Normalise write
        fname = fullfile(SubDir,'anat', sprintf('y_%s', aname(1).name));
        matlabbatch{2}.spm.spatial.normalise.write.subj.def = {fname};
        matlabbatch{2}.spm.spatial.normalise.write.subj.resample = func_4d;
        matlabbatch{2}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70; 78 76 85];
        matlabbatch{2}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
        matlabbatch{2}.spm.spatial.normalise.write.woptions.interp = 4;
        matlabbatch{2}.spm.spatial.normalise.write.woptions.prefix = 'w2';

        % Smooth EPI
        matlabbatch{3}.spm.spatial.smooth.data(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
        matlabbatch{3}.spm.spatial.smooth.fwhm = [6 6 6];
        matlabbatch{3}.spm.spatial.smooth.dtype = 0;
        matlabbatch{3}.spm.spatial.smooth.im = 0;
        matlabbatch{3}.spm.spatial.smooth.prefix = 's6';

        % Run preprocessing
        job_name = fullfile(jobdir, 'Post_PAID_Processing1.mat');
        save(job_name, 'matlabbatch');
        spm_jobman('run', matlabbatch)
        clear matlabbatch
    
    end
    
% move the spm ps file to current tedana dir
spm_ps_file = dir(fullfile(pwd,'spm_*.ps'));
if ~isempty(spm_ps_file)
    inputPS = spm_ps_file.name;
    outputPDF = sprintf('spm_post_PAID_%s.pdf',string(datetime));
    command = sprintf('%s -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile="%s" "%s"', ghostscriptPath, outputPDF, inputPS);
    system(command);
    movefile(fullfile(pwd,inputPS),jobdir)
    movefile(fullfile(pwd,outputPDF),jobdir)
end
