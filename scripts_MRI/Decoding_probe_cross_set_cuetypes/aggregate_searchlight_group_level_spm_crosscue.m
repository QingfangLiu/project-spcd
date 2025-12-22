% aggregate_searchlight_group_level_spm_crosscue.m
%
% - One-sample t-test in SPM on (accuracy - chance)
% - Explicit GM mask: /Users/liuq13/SPCD_data/masks/gm_0.1.nii
% - Skips subjects: sub-19, sub-33
% - Keeps results separate in .../group_stats_spm/session-<tag>/

addpath('/Users/liuq13/Applications/spm12/');

%% ---------- Settings ----------
base_dir   = '/Users/liuq13/SPCD_data/searchlight_maps';
subfolder  = 'probe_crosscue';

modes      = {'strict', 'set2set'};  % cross-cue variants
cue_fixed  = 'AB';                  % filenames use cue-AB
class_list = [2, 3];

session_tag = 'premeal';            % matches filename pattern
radius_str  = '8mm';
fwhm        = 6;                    % mm FWHM
group_stats_suffix = 'group_stats_spm';

% explicit GM mask
mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';

% subjects to skip
skip_subs = {'sub-19', 'sub-33'};

%% ---------- Init SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

searchlight_dir = fullfile(base_dir, subfolder);
chance_fun = @(n_cls) (n_cls==2)*0.5 + (n_cls==3)*(1/3);

%% ---------- Main loops ----------
for n_cls = class_list
    for mi = 1:numel(modes)
        mode = modes{mi};
        mode_tag = sprintf('crosscue_%s', mode); % filenames have mode-crosscue_strict / mode-crosscue_set2set

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
        keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), map_files_gz);
        map_files_gz = map_files_gz(keep_idx);

        fprintf('Found %d searchlight maps (after skipping sub-19, sub-33).\n', numel(map_files_gz));

        if isempty(map_files_gz)
            fprintf('No maps found. Skipping.\n');
            continue;
        end

        %% ---- Gunzip, apply GM mask, center at chance, smooth ----
        chance_val     = chance_fun(n_cls);
        smoothed_files = {};

        % Load GM mask once (assumes same space/dims as maps)
        Vm = spm_vol(mask_file);
        GM = spm_read_vols(Vm) > 0;

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
            [~, base_name, ~] = fileparts(nii_path);  % strips '.nii' from '...something.nii'
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
