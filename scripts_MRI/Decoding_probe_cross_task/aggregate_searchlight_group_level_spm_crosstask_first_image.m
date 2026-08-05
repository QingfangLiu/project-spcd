% Group SPM analyses for binary first-image cross-task searchlight maps.

addpath('/Users/liuq13/Applications/spm12/')

%% ---------- Settings ----------
base_dir = '/Users/liuq13/SPCD_data/searchlight_maps';
subfolder = 'probe_crosstask';
mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';

cue_subsets = {'conditioned_odor_only', 'preconditioned_odor_only'};
session_tag = 'premeal';
radius_str = '8mm';
fwhm = 6;
mode_tag = 'crosstask';
class_tag = 'classes-2';
image_tag = 'images-first';
chance_val = 0.5;
group_stats_suffix = 'group_stats_spm';

%% ---------- Initialize SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

searchlight_dir = fullfile(base_dir, subfolder);
M = spm_read_vols(spm_vol(mask_file)) > 0;

%% ---------- Analyze each cue subset separately ----------
for si = 1:numel(cue_subsets)
    cue_subset = cue_subsets{si};

    fprintf('\n>>> Binary first-image SPM | subset=%s | session=%s\n', ...
        cue_subset, session_tag);

    combo_dir = fullfile(searchlight_dir, ...
        sprintf('mode-%s_subset-%s_%s_%s', ...
                mode_tag, cue_subset, class_tag, image_tag));
    output_dir = fullfile(combo_dir, group_stats_suffix, ...
        sprintf('session-%s', session_tag));
    if ~exist(output_dir,'dir'), mkdir(output_dir); end

    spm_input_dir = fullfile(output_dir, 'spm_inputs');
    if ~exist(spm_input_dir,'dir'), mkdir(spm_input_dir); end

    %% --------- Find this subset's binary first-image maps ----------
    pattern = fullfile(searchlight_dir, ...
        sprintf(['sub-*_%s_mode-%s_subset-%s_%s_' ...
                 '%s_radius-%s.nii.gz'], ...
                ['session-' session_tag], mode_tag, cue_subset, ...
                class_tag, image_tag, radius_str));
    map_files_gz = dir(pattern);

    skip_subs = {'sub-19', 'sub-33'};
    keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), map_files_gz);
    map_files_gz = map_files_gz(keep_idx);

    fprintf('Found %d maps (after skipping sub-19, sub-33).\n', ...
            numel(map_files_gz));
    if isempty(map_files_gz), continue; end

    %% --------- Unzip, center at binary chance, and smooth ----------
    smoothed_files = {};
    for i = 1:numel(map_files_gz)
        gz_path = fullfile(map_files_gz(i).folder, map_files_gz(i).name);
        [~,name,~] = fileparts(gz_path);
        nii_path = fullfile(map_files_gz(i).folder, name);
        if ~exist(nii_path,'file')
            gunzip(gz_path, map_files_gz(i).folder);
        end

        V = spm_vol(nii_path);
        Y = spm_read_vols(V);
        Y(~M) = NaN;
        Yc = Y - chance_val;

        centered_path = fullfile(spm_input_dir, ['c_' name]);
        Vc = V;
        Vc.fname = centered_path;
        spm_write_vol(Vc, Yc);

        smooth_path = fullfile(spm_input_dir, ['s_' name]);
        spm_smooth(centered_path, smooth_path, [fwhm fwhm fwhm]);
        smoothed_files{end+1,1} = [smooth_path ',1'];
    end

    %% --------- SPM one-sample t-test ----------
    matlabbatch = {};
    matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = smoothed_files;
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 0;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {mask_file};

    matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(output_dir, 'SPM.mat')};
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    matlabbatch{3}.spm.stats.con.spmmat = {fullfile(output_dir, 'SPM.mat')};
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Decoding > chance';
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;

    spm_jobman('run', matlabbatch);
    fprintf('  -> Done. T-map: %s\n', fullfile(output_dir,'spmT_0001.nii'));
end

fprintf('\nBoth binary first-image cue subsets finished.\n');
