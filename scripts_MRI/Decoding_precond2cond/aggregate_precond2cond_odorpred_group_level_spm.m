% aggregate_precond2cond_odorpred_group_level_spm.m
%
% SPM12 group analysis for preconditioned→conditioned searchlight maps
% during the odor prediction task.
%
% - Loads subject-level maps:
%     sub-*_session-odorpred_mode-precond2cond_<glm_tag>_classes-6_radius-<radius>.nii.gz
% - Centers each map at chance (1/6), smooths with SPM
% - Runs a one-sample t-test in SPM on (acc - chance)
% - Uses explicit GM mask: base_dir/masks/gm_0.1.nii
% - Stores results in:
%     odorpred_crosstask/group_stats_spm/session-odorpred_glm-<glm_tag>/

addpath('/Users/liuq13/Applications/spm12/')

%% ---------- Settings (edit) ----------
% Base searchlight_maps dir (parent of odorpred_crosstask)
base_dir   = '/Users/liuq13/SPCD_data/searchlight_maps';
subfolder  = 'odorpred_crosstask';

% Session tag is fixed by our filenames
session_tag = 'odorpred';

% Choose which odor-prediction GLM variant to aggregate:
%   'cueEARLY_trialwise_2mm'  or  'cueEXT5s_trialwise_2mm'
% glm_tag    = 'cueEARLY_trialwise_2mm';   % <-- change to 'cueEXT5s_trialwise_2mm' if needed
glm_tag = 'cueEXT5s_trialwise_2mm';

radius_str = '8mm';       % must match your filenames
fwhm       = 6;           % mm FWHM
chance     = 1/6;         % 6-way classification

group_stats_suffix = 'group_stats_spm';

% explicit GM mask 
mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';

%% ---------- Init SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

if ~exist(mask_file, 'file')
    error('Explicit mask not found: %s', mask_file);
end

searchlight_dir = fullfile(base_dir, subfolder);

fprintf('\n>>> SPM group analysis | mode=precond2cond | classes=6 | session=%s | glm=%s\n', ...
    session_tag, glm_tag);

% Output directory (separate folder per GLM option)
output_dir = fullfile(searchlight_dir, group_stats_suffix, ...
                      sprintf('session-%s_glm-%s', session_tag, glm_tag));
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% Folder for centered & smoothed subject images
spm_input_dir = fullfile(output_dir, 'spm_inputs');
if ~exist(spm_input_dir, 'dir'), mkdir(spm_input_dir); end

%% ---------- Find subject maps (.nii.gz) ----------
pattern = fullfile(searchlight_dir, ...
    sprintf('sub-*_session-%s_mode-precond2cond_%s_classes-6_radius-%s.nii.gz', ...
            session_tag, glm_tag, radius_str));

map_files_gz = dir(pattern);
fprintf('Found %d searchlight maps.\n', numel(map_files_gz));
if isempty(map_files_gz)
    error('No maps found matching pattern:\n  %s', pattern);
end

%% ---------- Center at chance & smooth ----------
smoothed_files = {};

for i = 1:numel(map_files_gz)
    gz_path = fullfile(map_files_gz(i).folder, map_files_gz(i).name);

    % '.nii.gz' -> '.nii'
    [gz_folder, gz_name, gz_ext] = fileparts(gz_path);  % gz_ext = '.gz'
    if ~strcmp(gz_ext, '.gz')
        error('Expected .nii.gz file, got: %s', gz_path);
    end
    nii_path = fullfile(gz_folder, gz_name);  % includes '.nii' in gz_name

    % gunzip if needed
    if ~exist(nii_path, 'file')
        fprintf('  Gunzip: %s\n', gz_path);
        gunzip(gz_path, gz_folder);
    end

    % Load accuracy map
    V = spm_vol(nii_path);
    Y = spm_read_vols(V);

    % Center at chance: acc - chance
    Yc = Y - chance;

    % Save centered image
    [~, base_name, ~] = fileparts(nii_path);  % strip '.nii'
    centered_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
    Vc       = V;
    Vc.fname = centered_path;
    spm_write_vol(Vc, Yc);

    % Smooth centered image
    smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
    fprintf('  Center + smooth: %s -> %s\n', centered_path, smooth_path);
    spm_smooth(centered_path, smooth_path, [fwhm fwhm fwhm]);

    smoothed_files{end+1,1} = [smooth_path ',1'];
end

fprintf('Subjects processed: %d\n', numel(smoothed_files));

%% ---------- SPM one-sample t-test on (acc - chance) ----------
matlabbatch = {};

% Factorial design
matlabbatch{1}.spm.stats.factorial_design.dir                 = {output_dir};
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans        = smoothed_files;
matlabbatch{1}.spm.stats.factorial_design.cov                 = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none  = 1;

% Use explicit GM mask only (no implicit mask, no extra processing)
matlabbatch{1}.spm.stats.factorial_design.masking.im          = 0;
matlabbatch{1}.spm.stats.factorial_design.masking.em          = {mask_file};

matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit      = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm     = 1;

% Model estimation
spm_mat_path = fullfile(output_dir, 'SPM.mat');
matlabbatch{2}.spm.stats.fmri_est.spmmat                      = {spm_mat_path};
matlabbatch{2}.spm.stats.fmri_est.write_residuals             = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical            = 1;

% Contrast: Decoding > chance
matlabbatch{3}.spm.stats.con.spmmat                           = {spm_mat_path};
matlabbatch{3}.spm.stats.con.delete                           = 0;
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name             = 'Decoding > chance';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights          = 1;
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep          = 'none';

fprintf('Running SPM batch in: %s\n', output_dir);
spm_jobman('run', matlabbatch);

fprintf('\n✅ Done. Check t-map:\n  %s\n', fullfile(output_dir, 'spmT_0001.nii'));
