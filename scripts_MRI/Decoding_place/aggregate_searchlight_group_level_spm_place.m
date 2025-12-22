% aggregate_searchlight_group_level_spm_place.m
%
% - Input files: sub-*searchlight.nii.gz
% - Loop over normalization subfolders
% - Smooth (6mm), center at chance, one-sample t-test
% - Explicit GM mask

addpath('/Users/liuq13/Applications/spm12/');

%% ---------- Settings ----------
base_dir     = '/Users/liuq13/SPCD_data/searchlight_maps/place_scan';
norm_options = {'none', 'across_trials'};
chance_level = 1/6;
fwhm         = 6;

group_stats_suffix = 'group_stats_spm';
mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';

skip_subs = {'sub-19', 'sub-33'};

%% ---------- Init SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

%% ---------- Load GM mask ----------
Vm = spm_vol(mask_file);
GM = spm_read_vols(Vm) > 0;

%% ---------- Main loop ----------
for ni = 1:numel(norm_options)
    norm = norm_options{ni};
    fprintf('\n>>> Running group analysis for normalization: %s\n', norm);

    searchlight_dir = fullfile(base_dir, norm);
    output_dir      = fullfile(searchlight_dir, 'group_stats');
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end

    spm_input_dir = fullfile(output_dir, 'spm_inputs');
    if ~exist(spm_input_dir, 'dir'), mkdir(spm_input_dir); end

    %% ---- Find subject maps ----
    pattern = fullfile(searchlight_dir, 'sub-*searchlight.nii.gz');
    map_files_gz = dir(pattern);

    % skip subjects
    keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), map_files_gz);
    map_files_gz = map_files_gz(keep_idx);

    fprintf('Found %d searchlight maps\n', numel(map_files_gz));
    if isempty(map_files_gz)
        fprintf('No maps found. Skipping.\n');
        continue;
    end

    %% ---- Gunzip, mask, center, smooth ----
    smoothed_files = {};

    for i = 1:numel(map_files_gz)
        gz_path = fullfile(map_files_gz(i).folder, map_files_gz(i).name);
        [gz_folder, gz_name, gz_ext] = fileparts(gz_path);

        if ~strcmp(gz_ext, '.gz')
            error('Expected .nii.gz: %s', gz_path);
        end

        nii_path = fullfile(gz_folder, gz_name); % gz_name includes .nii

        if ~exist(nii_path, 'file')
            gunzip(gz_path, gz_folder);
        end

        % load map
        V = spm_vol(nii_path);
        Y = spm_read_vols(V);

        % apply GM mask
        if ~isequal(size(Y), size(GM))
            error('GM mask does not match map dims: %s', nii_path);
        end
        Y(~GM) = NaN;

        % center at chance
        Yc = Y - chance_level;

        % save centered
        [~, base_name, ~] = fileparts(nii_path);
        centered_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
        Vc = V; 
        Vc.fname = centered_path;
        spm_write_vol(Vc, Yc);

        % smooth
        smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
        spm_smooth(centered_path, smooth_path, [fwhm fwhm fwhm]);

        smoothed_files{end+1,1} = [smooth_path ',1'];
    end

    %% ---- One-sample t-test in SPM ----
    matlabbatch = {};

    matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = smoothed_files;

    matlabbatch{1}.spm.stats.factorial_design.cov = struct([]);

    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 0;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {mask_file};

    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

    spm_mat = fullfile(output_dir, 'SPM.mat');
    matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat};
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    matlabbatch{3}.spm.stats.con.spmmat = {spm_mat};
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Decoding > chance';
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

    fprintf('Running SPM batch: %s\n', output_dir);
    spm_jobman('run', matlabbatch);

    fprintf('  -> Done. T-map: %s\n', fullfile(output_dir, 'spmT_0001.nii'));
end

fprintf('\nAll normalizations finished.\n');
