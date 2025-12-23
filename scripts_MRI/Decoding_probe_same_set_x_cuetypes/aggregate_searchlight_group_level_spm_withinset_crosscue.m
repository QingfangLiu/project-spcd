
% aggregate_searchlight_group_level_spm_withinset_crosscue.m
%
% Group-level (SPM12) one-sample t-test on (accuracy - chance) for:
%   - within-set, cross-cue decoding (A <-> B)
%
% Steps per subject map:
%   1) gunzip .nii.gz -> .nii (if needed)
%   2) apply explicit GM mask (outside -> NaN)
%   3) center at chance: acc - chance
%   4) smooth centered maps (SPM smoothing)
%   5) one-sample t-test in SPM on centered maps
%
% Output:
%   <base_dir>/<subfolder>/mode-<mode_tag>_cue-AB_classes-<2|3>/group_stats_spm/session-<tag>/

addpath('/Users/liuq13/Applications/spm12/');

%% ---------- Settings ----------
base_dir   = '/Users/liuq13/SPCD_data/searchlight_maps';

% <-- IMPORTANT: set this to the folder where your new within-set crosscue maps live
subfolder  = 'probe_withinset_crosscue';

mode_tags  = { ...
    'withinset_strict', ...
    'withinset_set2set' ...
};

cue_fixed  = 'AB';     % filenames use cue-AB
class_list = [2, 3];

session_tag = 'premeal';   % e.g., premeal / postmeal
radius_str  = '8mm';
fwhm        = 6;           % smoothing kernel (mm)
group_stats_suffix = 'group_stats_spm';

% explicit GM mask
mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';

% subjects to skip (leave empty if not needed)
skip_subs = {'sub-19', 'sub-33'};

%% ---------- Init SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

searchlight_dir = fullfile(base_dir, subfolder);
chance_fun = @(n_cls) (n_cls==2)*0.5 + (n_cls==3)*(1/3);

%% ---------- Load GM mask once ----------
Vm = spm_vol(mask_file);
GM = spm_read_vols(Vm) > 0;

%% ---------- Main loops ----------
for n_cls = class_list
    for mi = 1:numel(mode_tags)
        mode_tag = mode_tags{mi};

        fprintf('\n>>> SPM group analysis | classes=%d | mode=%s | cue=%s | session=%s\n', ...
            n_cls, mode_tag, cue_fixed, session_tag);

        combo_dir  = fullfile(searchlight_dir, ...
            sprintf('mode-%s_cue-%s_classes-%d', mode_tag, cue_fixed, n_cls));

        output_dir = fullfile(combo_dir, group_stats_suffix, ...
            sprintf('session-%s', session_tag));

        if ~exist(output_dir, 'dir'), mkdir(output_dir); end

        spm_input_dir = fullfile(output_dir, 'spm_inputs');
        if ~exist(spm_input_dir, 'dir'), mkdir(spm_input_dir); end

        %% ---- Find subject maps (.nii.gz) ----
        pattern = fullfile(searchlight_dir, ...
            sprintf('sub-*_session-%s_mode-%s_cue-%s_classes-%d_radius-%s.nii.gz', ...
                    session_tag, mode_tag, cue_fixed, n_cls, radius_str));

        map_files_gz = dir(pattern);

        % filter out skipped subjects
        if ~isempty(skip_subs)
            keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), map_files_gz);
            map_files_gz = map_files_gz(keep_idx);
        end

        fprintf('Found %d searchlight maps (after skipping).\n', numel(map_files_gz));

        if isempty(map_files_gz)
            fprintf('No maps found for this combination. Skipping.\n');
            continue;
        end

        %% ---- Gunzip, apply GM mask, center at chance, smooth ----
        chance_val     = chance_fun(n_cls);
        smoothed_files = cell(numel(map_files_gz), 1);

        for i = 1:numel(map_files_gz)
            gz_path = fullfile(map_files_gz(i).folder, map_files_gz(i).name);

            % decompose 'xxx.nii.gz' -> 'xxx.nii'
            [gz_folder, gz_name, gz_ext] = fileparts(gz_path); % gz_ext = '.gz'
            if ~strcmp(gz_ext, '.gz')
                error('Expected .nii.gz file, got: %s', gz_path);
            end
            nii_path = fullfile(gz_folder, gz_name);  % gz_name includes '.nii'

            if ~exist(nii_path, 'file')
                fprintf('  Gunzip: %s\n', gz_path);
                gunzip(gz_path, gz_folder);
            end

            % Load accuracy map
            V = spm_vol(nii_path);
            Y = spm_read_vols(V);

            % Apply GM mask: outside mask -> NaN
            if ~isequal(size(Y), size(GM))
                error('GM mask dims (%s) do not match map dims (%s) for file: %s', ...
                    mat2str(size(GM)), mat2str(size(Y)), nii_path);
            end
            Y(~GM) = NaN;

            % Center at chance: acc - chance
            Yc = Y - chance_val;

            % Save centered image
            [~, base_name, ~] = fileparts(nii_path);  % strips '.nii'
            centered_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
            Vc       = V;
            Vc.fname = centered_path;
            spm_write_vol(Vc, Yc);

            % Smooth centered image
            smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
            fprintf('  Center + smooth: %s -> %s\n', centered_path, smooth_path);
            spm_smooth(centered_path, smooth_path, [fwhm fwhm fwhm]);

            smoothed_files{i,1} = [smooth_path ',1'];
        end

        %% ---- SPM one-sample t-test on (acc - chance) ----
        matlabbatch = {};

        % Factorial design
        matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir};
        matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = smoothed_files;

        % No covariates
        matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});

        % Explicit mask, no implicit mask
        matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
        matlabbatch{1}.spm.stats.factorial_design.masking.im = 0;
        matlabbatch{1}.spm.stats.factorial_design.masking.em = {mask_file};

        % Global options (defaults)
        matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

        % Model estimation
        spm_mat_path = fullfile(output_dir, 'SPM.mat');
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat_path};
        matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

        % Contrast: Decoding > chance
        matlabbatch{3}.spm.stats.con.spmmat = {spm_mat_path};
        matlabbatch{3}.spm.stats.con.delete = 0;
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = 'Decoding > chance';
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

        % Run batch
        fprintf('Running SPM batch in: %s\n', output_dir);
        spm_jobman('run', matlabbatch);

        fprintf('  -> Done. T-map: %s\n', fullfile(output_dir, 'spmT_0001.nii'));
    end
end

fprintf('\nAll combinations finished.\n');
