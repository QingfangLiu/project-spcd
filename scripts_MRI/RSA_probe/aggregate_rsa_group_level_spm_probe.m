
% aggregate_rsa_group_level_spm_probe.m
%
% Group-level SPM analysis for RSA searchlight beta maps.
%
% INPUT layout:
%   <base_dir>/<subfolder>/<session_tag>/
%       sub-01_session-premeal_model-association_radius-8mm.nii.gz
%       sub-01_session-premeal_model-identity_radius-8mm.nii.gz
%       ...
%
% OUTPUT layout:
%   <base_dir>/<subfolder>/group_stats_spm/session-<session_tag>/model-<model>/
%
% Core logic preserved from decoding script:
%   - gunzip .nii.gz -> .nii
%   - load GM mask
%   - set outside mask to NaN
%   - smooth
%   - SPM one-sample t-test
%
% Requires SPM12 on MATLAB path.

addpath('/Users/liuq13/Applications/spm12/');

%% ---------- Settings ----------
base_dir  = '/Users/liuq13/SPCD_data/rsa_maps';
subfolder = 'probe_searchlight';

session_tag = 'premeal';   % 'premeal' or 'postmeal'
radius_str  = '8mm';       % matches filename: radius-8mm
fwhm        = 6;           % smoothing kernel in mm
group_stats_suffix = 'group_stats_spm';

% explicit GM mask (used both for NaN conversion AND SPM explicit mask)
mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';

% two RSA models per subject
models = {'association', 'identity'};

% subject skip list
skip_subs = {'sub-19', 'sub-33'};

%% ---------- Init SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

searchlight_dir = fullfile(base_dir, subfolder);

% input folder contains maps
input_dir = fullfile(searchlight_dir, session_tag);
if ~exist(input_dir, 'dir')
    error('Input folder not found: %s', input_dir);
end

% output root
group_root = fullfile(searchlight_dir, group_stats_suffix, sprintf('session-%s', session_tag));
if ~exist(group_root, 'dir'), mkdir(group_root); end

% load GM mask once (logic preserved; moved out of inner loops)
M = spm_read_vols(spm_vol(mask_file)) > 0;

%% ---------- Helper: list subject maps for a given model ----------
get_model_files = @(model) dir(fullfile(input_dir, ...
    sprintf('sub-*_session-%s_model-%s_radius-%s.nii.gz', session_tag, model, radius_str)));

%% ---------- Main: one-sample t-test per model (beta > 0) ----------
for mi = 1:numel(models)
    model = models{mi};

    fprintf('\n>>> RSA SPM group analysis | model=%s | session=%s | radius=%s\n', ...
        model, session_tag, radius_str);

    output_dir = fullfile(group_root, sprintf('model-%s', model));
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end

    spm_input_dir = fullfile(output_dir, 'spm_inputs');
    if ~exist(spm_input_dir, 'dir'), mkdir(spm_input_dir); end

    % ---- Find subject maps (.nii.gz) ----
    map_files_gz = get_model_files(model);

    % filter skipped subjects
    keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), map_files_gz);
    map_files_gz = map_files_gz(keep_idx);

    fprintf('Found %d RSA maps (after skipping).\n', numel(map_files_gz));
    if isempty(map_files_gz)
        fprintf('No maps found. Skipping model=%s.\n', model);
        continue;
    end

    %% ---- Gunzip, mask->NaN, smooth ----
    smoothed_files = {};

    for i = 1:numel(map_files_gz)
        gz_path = fullfile(map_files_gz(i).folder, map_files_gz(i).name);

        % decompose 'xxx.nii.gz' -> 'xxx.nii'
        [gz_folder, gz_name, gz_ext] = fileparts(gz_path); % gz_ext = '.gz'
        if ~strcmp(gz_ext, '.gz')
            error('Expected .nii.gz file, got: %s', gz_path);
        end
        nii_path = fullfile(gz_folder, gz_name);  % gz_name includes ".nii"

        if ~exist(nii_path, 'file')
            fprintf('  Gunzip: %s\n', gz_path);
            gunzip(gz_path, gz_folder);
        end

        % Load beta map
        V = spm_vol(nii_path);
        Y = spm_read_vols(V);

        % Set outside mask to NaN (important logic)
        Y(~M) = NaN;

        % For RSA betas, test is beta > 0
        Yc = Y;

        % Save masked/centered image
        [~, base_name, ~] = fileparts(nii_path); % strips ".nii"
        centered_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
        Vc = V;
        Vc.fname = centered_path;
        spm_write_vol(Vc, Yc);

        % Smooth
        smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
        fprintf('  Mask->NaN + smooth: %s -> %s\n', centered_path, smooth_path);
        spm_smooth(centered_path, smooth_path, [fwhm fwhm fwhm]);

        smoothed_files{end+1,1} = [smooth_path ',1'];
    end

    %% ---- SPM one-sample t-test on beta ----
    matlabbatch = {};

    % Factorial design
    matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = smoothed_files;

    % No covariates
    matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});

    % Explicit mask, no implicit mask (preserved)
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 0;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {mask_file};

    % Global options (defaults; preserved)
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

    % Model estimation
    spm_mat_path = fullfile(output_dir, 'SPM.mat');
    matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat_path};
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    % Contrast: RSA beta > 0
    matlabbatch{3}.spm.stats.con.spmmat = {spm_mat_path};
    matlabbatch{3}.spm.stats.con.delete = 0;
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = sprintf('RSA %s > 0', model);
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

    % Run batch
    fprintf('Running SPM batch in: %s\n', output_dir);
    spm_jobman('run', matlabbatch);

    fprintf('  -> Done. T-map: %s\n', fullfile(output_dir, 'spmT_0001.nii'));
end