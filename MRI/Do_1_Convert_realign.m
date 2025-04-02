
spm fmri

%%    
% convert dicom to nifti files
% and put in folders "1" and "2"
for s = 1:2 % two scan sessions

    fromdir = fullfile(SubDir,sprintf('Scan%d',s));
    outdir = fullfile(SubDir,'nii',num2str(s));

    if ~exist(outdir,'dir')
        mkdir(outdir);
        system(sprintf('%s -o %s %s', dcm2niipath, outdir, fromdir))
    end
end

%%

% organize images to different tasks
% need the following: T1 MPRAGE, 'place', 'character', 'odor_scan'
% 'odorpred', 'premeal', 'postmeal'
% note: some fMRI scans also have localizer sequence but not using them

for k = 1:length(pattern_list)
    
    pattern = pattern_list{k};
    nii_folder = fullfile(SubDir,'nii');
    tmp_out_folder = fullfile(nii_folder, 'func', pattern);
    if ~exist(tmp_out_folder, 'dir')
        mkdir(tmp_out_folder);
    else
        continue; 
    end

    nifti_files = dir(fullfile(nii_folder, '**', ['*' pattern '*.nii']));
    nifti_files = nifti_files([nifti_files.bytes] > size_threshold);  % Filter the files based on the size limit (greater than 2 MB)

    % Copy all matching files to the individual folder
    for i = 1:length(nifti_files)
        source_file = fullfile(nifti_files(i).folder, nifti_files(i).name);
        destination_file = fullfile(tmp_out_folder, nifti_files(i).name);
        copyfile(source_file, destination_file);
    end
end

% organize anat scan
pattern = 'MPRAGE';
tmp_out_folder = fullfile(nii_folder, 'anat');
if ~exist(tmp_out_folder, 'dir')
    mkdir(tmp_out_folder);
    nifti_files = dir(fullfile(nii_folder, '**', ['*' pattern '*.nii']));
    nifti_files = nifti_files([nifti_files.bytes] > size_threshold);  % Filter the files based on the size limit (greater than 2 MB)
    for i = 1:length(nifti_files)
        source_file = fullfile(nifti_files(i).folder, nifti_files(i).name);
        destination_file = fullfile(tmp_out_folder, nifti_files(i).name);
        copyfile(source_file, destination_file);
    end
end

%%

% organize echo 1 images across runs and sessions
% note different runs have different volume numbers
    
n_e1 = dir(fullfile(SubDir,'nii', 'func', '*', '*_e1.nii')); 
nruns = length(n_e1);

% count # volumes in each run
nscans = zeros(nruns,1);
for r = 1:nruns
    tmp_file = fullfile(n_e1(r).folder, n_e1(r).name);
    nscans(r) = length(spm_vol(tmp_file));
end

filename = [];
for r = 1:nruns
    tmp_file = fullfile(n_e1(r).folder, n_e1(r).name);
for i = 1:nscans(r)
    filename{r}{i,1} = sprintf('%s,%01d',tmp_file, i);
end
end

%%

disp('Realign using e1 data from all runs and sessions.')
matlabbatch = [];
matlabbatch{1}.spm.spatial.realign.estimate.data = filename;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.quality = 0.9;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.sep = 4;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.fwhm = 5;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.rtm = 0; % register to first to be faster
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.interp = 2;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.weight = '';

% save and run job
fname = fullfile(jobdir, 'me_realignment_e1_all_runs_sessions.mat');
save(fname, 'matlabbatch');
spm_jobman('run', matlabbatch);

% move the spm ps file to current nii directory & convert to pdf
spm_ps_file = dir(fullfile(pwd,'spm_*.ps'));
if ~isempty(spm_ps_file)
    inputPS = spm_ps_file.name;
    outputPDF = 'spm_realign.pdf';
    
    command = sprintf('%s -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile="%s" "%s"', ghostscriptPath, outputPDF, inputPS);
    system(command);
    movefile(fullfile(pwd,inputPS),jobdir)
    movefile(fullfile(pwd,outputPDF),jobdir)
end

%%
% apply the voxel-to-world mapping from the realigned echo 1 to other echoes by updating header files
for r = 1:nruns
    tmp_file = fullfile(n_e1(r).folder, n_e1(r).name);
    for i = 1:nscans(r)
        V = spm_get_space(sprintf('%s,%01d', tmp_file, i));
        for iecho = 2:nechos
            curr_echo_file_name = strrep(n_e1(r).name, 'e1', ['e', num2str(iecho)]);
            spm_get_space(sprintf('%s,%01d', fullfile(n_e1(r).folder, curr_echo_file_name),i),V);
        end
    end
end