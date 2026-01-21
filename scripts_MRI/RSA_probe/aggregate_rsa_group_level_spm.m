
% Group-level SPM analysis for RSA searchlight maps with updated filenames.
%
% INPUT layout:
%   <base_dir>/<subfolder>/<session_tag>/
%       sub-01_session-premeal_association_kendall_tau_ns-corr_radius-8mm.nii.gz
%       sub-01_session-premeal_association_regression_beta_ns-euc_radius-8mm.nii.gz
%       sub-01_session-premeal_identity_spearman_rho_ns-corr_radius-8mm.nii.gz
%       sub-01_session-premeal_value_spearman_rho_ns-corr_radius-8mm.nii.gz
%       ...
%   and likewise for session_tag = postmeal
%
% OUTPUT layout:
%   <base_dir>/<subfolder>/<group_stats_suffix>/session-<session_tag>/<family>/<stat>/<metric>/
%
% Core logic preserved:
%   - gunzip .nii.gz -> .nii
%   - load GM mask once
%   - set outside mask to NaN
%   - smooth
%   - SPM one-sample t-test
%   - one-sided contrast (> 0) for ALL maps (per user)
%
% Requires SPM12 on MATLAB path.

addpath('/Users/liuq13/Applications/spm12/');

%% ---------- Settings ----------
base_dir  = '/Users/liuq13/SPCD_data/rsa_maps';
subfolder = 'probe';

% Run both sessions automatically
sessions = {'premeal', 'postmeal'};

radius_str  = '8mm';    % always 8mm; matches: radius-8mm
fwhm        = 6;        % smoothing kernel in mm
group_stats_suffix = 'group_stats_spm';

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

    %% ---- Discover all maps in this session folder (radius fixed) ----
    % Pattern based on your updated naming
    pat = sprintf('sub-*_session-%s_*_radius-%s.nii.gz', session_tag, radius_str);
    all_gz = dir(fullfile(input_dir, pat));

    % Filter skip_subs
    keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), all_gz);
    all_gz = all_gz(keep_idx);

    fprintf('Discovered %d .nii.gz maps in %s (after skipping).\n', numel(all_gz), input_dir);
    if isempty(all_gz)
        continue;
    end

    %% ---- Parse filenames into groups: (family, stat, metric) ----
    % Expected filename shape:
    % sub-01_session-premeal_<family>_<stat>_<metric>_radius-8mm.nii.gz
    %
    % Notes:
    % - <stat> can include underscores (kendall_tau, spearman_rho, regression_beta, etc.)
    % - <metric> is assumed to be ns-corr or ns-euc (contains a dash)
    %
    % Strategy:
    % 1) remove prefix "sub-XX_session-<tag>_" and suffix "_radius-8mm.nii.gz"
    % 2) split remainder by underscores
    % 3) family = token{1}
    % 4) metric = last token
    % 5) stat = join tokens 2..end-1 with underscores

    groups = struct();  % dynamic struct keyed by a safe string key
    group_keys = {};    % preserve ordering

    for i = 1:numel(all_gz)
        fname = all_gz(i).name;

        % strip prefix + suffix
        prefix = sprintf('sub-');
        % robust prefix strip via regexp
        core = regexprep(fname, ...
            sprintf('^sub-\\d+_session-%s_', session_tag), '');
        core = regexprep(core, ...
            sprintf('_radius-%s\\.nii\\.gz$', radius_str), '');

        % split core
        toks = strsplit(core, '_');
        if numel(toks) < 3
            fprintf('WARNING: cannot parse (too few tokens), skipping: %s\n', fname);
            continue;
        end

        family = toks{1};
        metric = toks{end};
        stat_tokens = toks(2:end-1);
        stat = strjoin(stat_tokens, '_');

        % Basic sanity: metric should start with "ns-"
        if ~startsWith(metric, 'ns-')
            fprintf('WARNING: metric token does not look like "ns-*", skipping: %s\n', fname);
            continue;
        end

        % key for grouping
        key = sprintf('%s__%s__%s', family, stat, metric);

        if ~isfield(groups, matlab.lang.makeValidName(key))
            gname = matlab.lang.makeValidName(key);
            groups.(gname).family = family;
            groups.(gname).stat   = stat;
            groups.(gname).metric = metric;
            groups.(gname).files  = {};
            group_keys{end+1} = gname; 
        end

        gname = matlab.lang.makeValidName(key);
        groups.(gname).files{end+1,1} = fullfile(all_gz(i).folder, all_gz(i).name);
    end

    fprintf('Parsed into %d groups (family × stat × metric).\n', numel(group_keys));
    if isempty(group_keys)
        continue;
    end

    %% ---- Run one-sample t-test per group ----
    for gi = 1:numel(group_keys)
        gname = group_keys{gi};
        family = groups.(gname).family;
        stat   = groups.(gname).stat;
        metric = groups.(gname).metric;
        files_gz = groups.(gname).files;

        fprintf('\n>>> GROUP: family=%s | stat=%s | metric=%s | session=%s | radius=%s\n', ...
            family, stat, metric, session_tag, radius_str);

        % output_dir: <group_root>/<family>/<stat>/<metric>/
        output_dir = fullfile(group_root, family, stat, metric);
        ensure_dir(output_dir);

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

            % Set outside mask to NaN (preserved)
            Y(~M) = NaN;

            % One-sided test (>0) for all maps: no centering needed
            Yc = Y;

            % Save masked image
            [~, base_name, ~] = fileparts(nii_path); % strips ".nii"
            masked_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
            Vc = V;
            Vc.fname = masked_path;
            spm_write_vol(Vc, Yc);

            % Smooth
            smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
            fprintf('  Mask->NaN + smooth: %s -> %s\n', masked_path, smooth_path);
            spm_smooth(masked_path, smooth_path, [fwhm fwhm fwhm]);

            smoothed_files{end+1,1} = [smooth_path ',1']; %#ok<SAGROW>
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

        % Explicit mask, no implicit mask (preserved)
        matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
        matlabbatch{1}.spm.stats.factorial_design.masking.im = 0;
        matlabbatch{1}.spm.stats.factorial_design.masking.em = {mask_file};

        % Global options (preserved)
        matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

        % Model estimation
        spm_mat_path = fullfile(output_dir, 'SPM.mat');
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat_path};
        matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

        % Contrast: map > 0 (one-sided, per user)
        matlabbatch{3}.spm.stats.con.spmmat = {spm_mat_path};
        matlabbatch{3}.spm.stats.con.delete = 0;

        con_name = sprintf('RSA %s | %s | %s > 0', family, stat, metric);
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
