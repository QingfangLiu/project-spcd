
% Group-level SPM analysis for RSA searchlight maps (subset-conditioned / subset-preconditioned).
%
% INPUT layout:
%   <base_dir>/<subfolder>/<session_tag>/
%       sub-01_session-postmeal_identity_spearman_rho_ns-corr_radius-8mm_subset-conditioned.nii.gz
%       sub-01_session-postmeal_identity_spearman_rho_ns-corr_radius-8mm_subset-preconditioned.nii.gz
%       sub-01_session-postmeal_value_spearman_rho_ns-corr_radius-8mm_subset-conditioned.nii.gz
%       sub-01_session-postmeal_value_spearman_rho_ns-corr_radius-8mm_subset-preconditioned.nii.gz
%       ... and likewise for premeal
%
% OUTPUT layout (simplified; stat/metric fixed):
%   <base_dir>/<subfolder>/<group_stats_suffix>/session-<session_tag>/<subset>/<family>/
%       SPM.mat, contrasts, etc.
%       spm_inputs/ (masked + smoothed subject maps used as inputs)
%
% Core logic preserved:
%   - gunzip .nii.gz -> .nii (if needed)
%   - load GM mask once
%   - set outside mask to NaN
%   - smooth
%   - SPM one-sample t-test
%   - one-sided contrast (> 0) for ALL maps
%
% Requires SPM12 on MATLAB path.

addpath('/Users/liuq13/Applications/spm12/');

%% ---------- Settings ----------
base_dir  = '/Users/liuq13/SPCD_data/rsa_maps';
subfolder = 'probe_subset';

sessions = {'premeal', 'postmeal'};

radius_str  = '8mm';    % fixed
fwhm        = 6;        % smoothing kernel (mm)
group_stats_suffix = 'group_stats_spm';

% Fixed tokens (simplify parsing / grouping)
fixed_stat   = 'spearman_rho';
fixed_metric = 'ns-corr';

% explicit GM mask (used both for NaN conversion AND SPM explicit mask)
mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';

% subject skip list
skip_subs = {'sub-19', 'sub-33'};

%% ---------- Init SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

searchlight_dir = fullfile(base_dir, subfolder);

% load GM mask once
M = spm_read_vols(spm_vol(mask_file)) > 0;

%% ---------- Helper: ensure directory ----------
ensure_dir = @(d) (exist(d,'dir') || mkdir(d));

%% ---------- Main loop over sessions ----------
for si = 1:numel(sessions)
    session_tag = sessions{si};

    fprintf('\n============================================================\n');
    fprintf('>>> SESSION: %s\n', session_tag);
    fprintf('============================================================\n');

    % input folder contains maps
    input_dir = fullfile(searchlight_dir, session_tag);
    if ~exist(input_dir, 'dir')
        fprintf('Input folder not found, skipping: %s\n', input_dir);
        continue;
    end

    % output root
    group_root = fullfile(searchlight_dir, group_stats_suffix, sprintf('session-%s', session_tag));
    ensure_dir(group_root);

    %% ---- Discover subset maps in this session folder ----
    % Only subset files exist, so match only:
    % sub-*_session-<tag>_*_radius-8mm_subset-*.nii.gz
    pat = sprintf('sub-*_session-%s_*_radius-%s_subset-*.nii.gz', session_tag, radius_str);
    all_gz = dir(fullfile(input_dir, pat));

    % Filter skip_subs
    keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), all_gz);
    all_gz = all_gz(keep_idx);

    fprintf('Discovered %d subset .nii.gz maps in %s (after skipping).\n', numel(all_gz), input_dir);
    if isempty(all_gz)
        continue;
    end

    %% ---- Parse filenames into groups: (family, subset) ----
    % Fixed: stat = spearman_rho, metric = ns-corr, radius = 8mm
    % Only need:
    %   family = first token after session prefix (identity/value)
    %   subset = token that starts with "subset-" (expected last token)

    groups = struct();   % keyed by safe name
    group_keys = {};     % preserve order

    for i = 1:numel(all_gz)
        fname = all_gz(i).name;

        % Sanity checks for fixed tokens (fail fast on wrong files)
        if ~contains(fname, ['_' fixed_stat '_'])
            fprintf('WARNING: missing expected stat token (%s), skipping: %s\n', fixed_stat, fname);
            continue;
        end
        if ~contains(fname, ['_' fixed_metric '_'])
            fprintf('WARNING: missing expected metric token (%s), skipping: %s\n', fixed_metric, fname);
            continue;
        end
        if ~contains(fname, ['_radius-' radius_str '_'])
            fprintf('WARNING: missing expected radius token (radius-%s), skipping: %s\n', radius_str, fname);
            continue;
        end

        % Strip prefix: sub-XX_session-<tag>_
        core = regexprep(fname, sprintf('^sub-\\d+_session-%s_', session_tag), '');
        % Strip suffix: .nii.gz
        core = regexprep(core, '\\.nii\\.gz$', '');

        toks = strsplit(core, '_');
        if numel(toks) < 2
            fprintf('WARNING: cannot parse (too few tokens), skipping: %s\n', fname);
            continue;
        end

        family = toks{1};

        % subset token expected: startsWith('subset-')
        subset_tok = '';
        for tt = 1:numel(toks)
            if startsWith(toks{tt}, 'subset-')
                subset_tok = toks{tt};
                break;
            end
        end

        if isempty(subset_tok)
            fprintf('WARNING: cannot find subset-* token, skipping: %s\n', fname);
            continue;
        end

        % Group key: family + subset (stat/metric fixed)
        key = sprintf('%s__%s', family, subset_tok);
        gname = matlab.lang.makeValidName(key);

        if ~isfield(groups, gname)
            groups.(gname).family = family;
            groups.(gname).subset = subset_tok;
            groups.(gname).files  = {};
            group_keys{end+1} = gname; %#ok<SAGROW>
        end

        groups.(gname).files{end+1,1} = fullfile(all_gz(i).folder, all_gz(i).name); %#ok<SAGROW>
    end

    fprintf('Parsed into %d groups (family × subset).\n', numel(group_keys));
    if isempty(group_keys)
        continue;
    end

    %% ---- Run one-sample t-test per group ----
    for gi = 1:numel(group_keys)
        gname = group_keys{gi};
        family = groups.(gname).family;
        subset = groups.(gname).subset;
        files_gz = groups.(gname).files;

        fprintf('\n>>> GROUP: session=%s | family=%s | subset=%s\n', session_tag, family, subset);

        % output_dir: <group_root>/<subset>/<family>/
        output_dir = fullfile(group_root, subset, family);
        ensure_dir(output_dir);

        % Skip if already done
        spm_mat_path = fullfile(output_dir, 'SPM.mat');
        if exist(spm_mat_path, 'file')
            fprintf('SPM.mat exists -> skipping group: %s\n', output_dir);
            continue;
        end

        spm_input_dir = fullfile(output_dir, 'spm_inputs');
        ensure_dir(spm_input_dir);

        fprintf('Found %d subject maps in this group.\n', numel(files_gz));
        if isempty(files_gz)
            continue;
        end

        %% ---- Gunzip, mask->NaN, smooth ----
        smoothed_files = cell(0,1);

        for i = 1:numel(files_gz)
            gz_path = files_gz{i};

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

            % Load map
            V = spm_vol(nii_path);
            Y = spm_read_vols(V);

            % Set outside mask to NaN
            Y(~M) = NaN;

            % Save masked image
            [~, base_name, ~] = fileparts(nii_path); % strips ".nii"
            masked_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
            Vc = V;
            Vc.fname = masked_path;
            spm_write_vol(Vc, Y);

            % Smooth
            smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
            fprintf('  Mask->NaN + smooth: %s -> %s\n', masked_path, smooth_path);
            spm_smooth(masked_path, smooth_path, [fwhm fwhm fwhm]);

            smoothed_files{end+1,1} = [smooth_path ',1'];
        end

        if isempty(smoothed_files)
            fprintf('No usable smoothed files produced; skipping group.\n');
            continue;
        end

        %% ---- SPM one-sample t-test ----
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

        % Global options
        matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

        % Model estimation
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat_path};
        matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

        % Contrast: map > 0 (one-sided)
        matlabbatch{3}.spm.stats.con.spmmat = {spm_mat_path};
        matlabbatch{3}.spm.stats.con.delete = 0;

        con_name = sprintf('RSA %s | %s | %s | %s > 0', family, fixed_stat, fixed_metric, subset);
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = con_name;
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

        % Run batch
        fprintf('Running SPM batch in: %s\n', output_dir);
        spm_jobman('run', matlabbatch);

        fprintf('  -> Done. T-map: %s\n', fullfile(output_dir, 'spmT_0001.nii'));
    end
end

fprintf('\nAll done.\n');
