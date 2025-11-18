% aggregate_cuereactivation_group_level_spm.m
%
% SPM12 group analysis for cue reactivation searchlight maps.
% - Loads subject-level maps:
%     sub-*_session-<session_tag>_mode-cond2precond_classes-6_radius-<radius>.nii.gz
% - Centers each map at chance (1/6), smooths with SPM
% - Runs a one-sample t-test in SPM on (acc - chance)
% - Uses explicit GM mask: base_dir/masks/gm_0.1.nii
% - Stores results in: cue_reactivation/group_stats_spm/session-<tag>/

%% ---------- Settings (edit) ----------
base_dir   = '/Users/liuq13/SPCD_data/searchlight_maps';
subfolder  = 'cue_reactivation';

session_tag = 'postmeal';   % must match your filenames
radius_str  = '8mm';       % must match your filenames
fwhm        = 6;           % mm FWHM
chance      = 1/6;         % 6-way classification

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

fprintf('\n>>> SPM group analysis | mode=cond2precond | classes=6 | session=%s\n', session_tag);

% Output directory
output_dir = fullfile(searchlight_dir, group_stats_suffix, ...
                      sprintf('session-%s', session_tag));
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% Folder for centered & smoothed subject images
spm_input_dir = fullfile(output_dir, 'spm_inputs');
if ~exist(spm_input_dir, 'dir'), mkdir(spm_input_dir); end

%% ---------- Find subject maps (.nii.gz) ----------
pattern = fullfile(searchlight_dir, ...
    sprintf(['sub-*_session-%s_mode-cond2precond_classes-6_radius-%s.nii.gz'], ...
            session_tag, radius_str));

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
matlabbatch{1}.spm.stats.factorial_design.dir              = {output_dir};
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans     = smoothed_files;
matlabbatch{1}.spm.stats.factorial_design.cov              = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;

% Use explicit GM mask only (no implicit mask, no extra processing)
matlabbatch{1}.spm.stats.factorial_design.masking.im       = 0;
matlabbatch{1}.spm.stats.factorial_design.masking.em       = {mask_file};

matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit   = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm  = 1;

% Model estimation
spm_mat_path = fullfile(output_dir, 'SPM.mat');
matlabbatch{2}.spm.stats.fmri_est.spmmat        = {spm_mat_path};
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

% Contrast: Decoding > chance
matlabbatch{3}.spm.stats.con.spmmat                         = {spm_mat_path};
matlabbatch{3}.spm.stats.con.delete                         = 0;
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name           = 'Decoding > chance';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights        = 1;
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep        = 'none';

fprintf('Running SPM batch in: %s\n', output_dir);
spm_jobman('run', matlabbatch);

fprintf('\n✅ Done. Check t-map:\n  %s\n', fullfile(output_dir, 'spmT_0001.nii'));
