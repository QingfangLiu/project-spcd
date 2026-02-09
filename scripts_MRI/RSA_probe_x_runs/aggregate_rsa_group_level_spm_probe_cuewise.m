
% aggregate_rsa_group_level_spm_probe_cuewise.m
%
% Group-level SPM analysis for RSA searchlight maps (probe_cuewise layout).
%
% INPUT layout:
%   <base_dir>/probe_cuewise/<session>/<input-data>/
%       sub-01_postmeal_input-beta_mode-both_identity_spearman_radius-8mm.nii.gz
%       sub-01_postmeal_input-tmap_mode-img1only_value_spearman_radius-8mm.nii.gz
%       ...
%   where:
%     session   = premeal | postmeal
%     input     = input-beta | input-tmap
%     mode      = both | img1only | img2only
%     model     = identity | value
%
% OUTPUT layout:
%   <base_dir>/probe_cuewise/group_stats_spm/
%       session-<session>/<input>/<mode>/<model>/
%       session-collapsed/<input>/<mode>/<model>/
%
% Core logic reused:
%   - gunzip .nii.gz -> .nii
%   - load GM mask once
%   - set outside mask to NaN
%   - smooth
%   - SPM one-sample t-test
%   - one-sided contrast (> 0) for ALL maps
%   - skip if SPM.mat already exists
%
% Requires SPM12 on MATLAB path.

%% ---------- Settings ----------
addpath('/Users/liuq13/Applications/spm12/');

base_dir  = '/Users/liuq13/SPCD_data/rsa_maps';
analysis_subfolder = 'probe_cuewise';

sessions = {'premeal', 'postmeal'};
inputs   = {'input-beta', 'input-tmap'};
modes    = {'both', 'img1only', 'img2only'};
models   = {'identity', 'value'};

radius_str = '8mm';  % matches: radius-8mm
fwhm       = 6;      % smoothing kernel in mm

group_stats_suffix = 'group_stats_spm';

% explicit GM mask (used both for NaN conversion AND SPM explicit mask)
mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';

% subject skip list (substring match)
skip_subs = {'sub-19', 'sub-33'};

% collapsed session tag
collapsed_tag = 'collapsed';

%% ---------- Init SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

analysis_root = fullfile(base_dir, analysis_subfolder);

% load GM mask once
M = spm_read_vols(spm_vol(mask_file)) > 0;

%% ---------- Helper: ensure directory ----------
ensure_dir = @(d) (exist(d,'dir') || mkdir(d));

%% ---------- Helper: gunzip if needed ----------
function nii_path = ensure_nii_from_niigz(gz_path)
    [gz_folder, gz_name, gz_ext] = fileparts(gz_path); % gz_ext='.gz'
    if ~strcmp(gz_ext, '.gz')
        error('Expected .nii.gz file, got: %s', gz_path);
    end
    nii_path = fullfile(gz_folder, gz_name); % gz_name includes ".nii"
    if ~exist(nii_path, 'file')
        gunzip(gz_path, gz_folder);
    end
end

%% ---------- Helper: write masked (NaN outside GM), then smooth ----------
function smooth_file_with_vol(V, Y, M, spm_input_dir, base_name, fwhm)
    % base_name should be without extension; output uses c_ and s_ prefixes.
    Y(~M) = NaN;

    masked_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
    Vc = V;
    Vc.fname = masked_path;
    spm_write_vol(Vc, Y);

    smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
    spm_smooth(masked_path, smooth_path, [fwhm fwhm fwhm]);
end

%% ---------- Part 1: Per-session group analysis ----------
for si = 1:numel(sessions)
    session_tag = sessions{si};

    fprintf('\n============================================================\n');
    fprintf('>>> PER-SESSION GROUP ANALYSIS | SESSION: %s\n', session_tag);
    fprintf('============================================================\n');

    for ii = 1:numel(inputs)
        input_tag = inputs{ii};

        input_dir = fullfile(analysis_root, session_tag, input_tag);
        if ~exist(input_dir, 'dir')
            fprintf('Input folder not found, skipping: %s\n', input_dir);
            continue;
        end

        for mi = 1:numel(modes)
            mode_tag = modes{mi};

            for modeli = 1:numel(models)
                model_tag = models{modeli};

                % Output dir:
                output_dir = fullfile(analysis_root, group_stats_suffix, ...
                    sprintf('session-%s', session_tag), input_tag, mode_tag, model_tag);
                ensure_dir(output_dir);

                % Skip if already done
                spm_mat_path = fullfile(output_dir, 'SPM.mat');
                if exist(spm_mat_path, 'file')
                    fprintf('SPM.mat exists, skipping: %s\n', output_dir);
                    continue;
                end

                spm_input_dir = fullfile(output_dir, 'spm_inputs');
                ensure_dir(spm_input_dir);

                % File pattern:
                % sub-01_postmeal_input-tmap_mode-both_identity_spearman_radius-8mm.nii.gz
                pat = sprintf('sub-*_%s_%s_mode-%s_%s_*_radius-%s.nii.gz', ...
                    session_tag, input_tag, mode_tag, model_tag, radius_str);

                all_gz = dir(fullfile(input_dir, pat));

                % Filter skip_subs
                keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), all_gz);
                all_gz = all_gz(keep_idx);

                fprintf('\n>>> GROUP: session=%s | input=%s | mode=%s | model=%s\n', ...
                    session_tag, input_tag, mode_tag, model_tag);
                fprintf('Found %d .nii.gz maps.\n', numel(all_gz));

                if isempty(all_gz)
                    continue;
                end

                % Build smoothed file list
                smoothed_files = cell(0,1);

                for fi = 1:numel(all_gz)
                    gz_path = fullfile(all_gz(fi).folder, all_gz(fi).name);

                    nii_path = ensure_nii_from_niigz(gz_path);

                    V = spm_vol(nii_path);
                    Y = spm_read_vols(V);

                    [~, base_name, ~] = fileparts(nii_path); % strips ".nii"
                    % Keep base_name unique; it already contains subject+tokens.
                    fprintf('  Mask->NaN + smooth: %s\n', base_name);

                    % Write masked + smooth
                    smooth_file_with_vol(V, Y, M, spm_input_dir, base_name, fwhm);

                    smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
                    smoothed_files{end+1,1} = [smooth_path ',1']; %#ok<SAGROW>
                end

                if isempty(smoothed_files)
                    fprintf('No usable smoothed files produced; skipping.\n');
                    continue;
                end

                % ---------- SPM one-sample t-test ----------
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

                con_name = sprintf('RSA %s | %s | %s | %s > 0', session_tag, input_tag, mode_tag, model_tag);
                matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = con_name;
                matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
                matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

                fprintf('Running SPM batch in: %s\n', output_dir);
                spm_jobman('run', matlabbatch);

                fprintf('  -> Done. spmT: %s\n', fullfile(output_dir, 'spmT_0001.nii'));
            end
        end
    end
end

%% ---------- Part 2: Collapsed across sessions (within-subject averaging) ----------
fprintf('\n============================================================\n');
fprintf('>>> COLLAPSED (WITHIN-SUBJECT AVERAGE) GROUP ANALYSIS\n');
fprintf('============================================================\n');

for ii = 1:numel(inputs)
    input_tag = inputs{ii};

    for mi = 1:numel(modes)
        mode_tag = modes{mi};

        for modeli = 1:numel(models)
            model_tag = models{modeli};

            % Output dir:
            output_dir = fullfile(analysis_root, group_stats_suffix, ...
                sprintf('session-%s', collapsed_tag), input_tag, mode_tag, model_tag);
            ensure_dir(output_dir);

            % Skip if already done
            spm_mat_path = fullfile(output_dir, 'SPM.mat');
            if exist(spm_mat_path, 'file')
                fprintf('SPM.mat exists, skipping: %s\n', output_dir);
                continue;
            end

            spm_input_dir = fullfile(output_dir, 'spm_inputs');
            ensure_dir(spm_input_dir);

            % Input dirs for both sessions
            pre_dir  = fullfile(analysis_root, 'premeal',  input_tag);
            post_dir = fullfile(analysis_root, 'postmeal', input_tag);

            if ~exist(pre_dir, 'dir') || ~exist(post_dir, 'dir')
                fprintf('Missing premeal/postmeal input dirs for %s, skipping.\n', input_tag);
                continue;
            end

            % Patterns per session
            pre_pat  = sprintf('sub-*_%s_%s_mode-%s_%s_*_radius-%s.nii.gz', ...
                'premeal', input_tag, mode_tag, model_tag, radius_str);
            post_pat = sprintf('sub-*_%s_%s_mode-%s_%s_*_radius-%s.nii.gz', ...
                'postmeal', input_tag, mode_tag, model_tag, radius_str);

            pre_gz  = dir(fullfile(pre_dir,  pre_pat));
            post_gz = dir(fullfile(post_dir, post_pat));

            % Filter skip_subs
            pre_keep  = ~arrayfun(@(f) any(contains(f.name, skip_subs)), pre_gz);
            post_keep = ~arrayfun(@(f) any(contains(f.name, skip_subs)), post_gz);
            pre_gz  = pre_gz(pre_keep);
            post_gz = post_gz(post_keep);

            fprintf('\n>>> COLLAPSED GROUP: input=%s | mode=%s | model=%s\n', ...
                input_tag, mode_tag, model_tag);
            fprintf('Premeal files: %d | Postmeal files: %d\n', numel(pre_gz), numel(post_gz));

            if isempty(pre_gz) || isempty(post_gz)
                fprintf('Not enough files to collapse; skipping.\n');
                continue;
            end

            % Build subject->file maps
            pre_map  = containers.Map();
            post_map = containers.Map();

            for k = 1:numel(pre_gz)
                nm = pre_gz(k).name;
                % subject id assumed as "sub-XX_" at start
                tok = regexp(nm, '^(sub-\d+)_', 'tokens', 'once');
                if isempty(tok); continue; end
                pre_map(tok{1}) = fullfile(pre_gz(k).folder, pre_gz(k).name);
            end

            for k = 1:numel(post_gz)
                nm = post_gz(k).name;
                tok = regexp(nm, '^(sub-\d+)_', 'tokens', 'once');
                if isempty(tok); continue; end
                post_map(tok{1}) = fullfile(post_gz(k).folder, post_gz(k).name);
            end

            % Intersection of subjects
            pre_keys = pre_map.keys;
            subj_keys = {};
            for k = 1:numel(pre_keys)
                s = pre_keys{k};
                if isKey(post_map, s)
                    subj_keys{end+1} = s; 
                end
            end
            subj_keys = sort(subj_keys);

            fprintf('Subjects with both sessions: %d\n', numel(subj_keys));
            if isempty(subj_keys)
                fprintf('No overlapping subjects; skipping.\n');
                continue;
            end

            % Create averaged (collapsed) images per subject, then smooth, then group t-test
            smoothed_files = cell(0,1);

            for k = 1:numel(subj_keys)
                sid = subj_keys{k};
                pre_gz_path  = pre_map(sid);
                post_gz_path = post_map(sid);

                pre_nii  = ensure_nii_from_niigz(pre_gz_path);
                post_nii = ensure_nii_from_niigz(post_gz_path);

                Vpre  = spm_vol(pre_nii);
                Vpost = spm_vol(post_nii);

                Ypre  = spm_read_vols(Vpre);
                Ypost = spm_read_vols(Vpost);

                % Apply GM mask to NaN first
                Ypre(~M)  = NaN;
                Ypost(~M) = NaN;

                % Within-subject average (NaN-safe)
                Yavg = (Ypre + Ypost) ./ 2;

                % Write averaged image using Vpre as template
                base_name = sprintf('%s_session-%s_%s_mode-%s_%s_radius-%s', ...
                    sid, collapsed_tag, input_tag, mode_tag, model_tag, radius_str);

                masked_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
                Vc = Vpre;
                Vc.fname = masked_path;
                spm_write_vol(Vc, Yavg);

                smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
                fprintf('  Collapsed avg + smooth: %s\n', base_name);
                spm_smooth(masked_path, smooth_path, [fwhm fwhm fwhm]);

                smoothed_files{end+1,1} = [smooth_path ',1']; %#ok<SAGROW>
            end

            if isempty(smoothed_files)
                fprintf('No usable collapsed smoothed files; skipping.\n');
                continue;
            end

            % ---------- SPM one-sample t-test ----------
            matlabbatch = {};

            matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir};
            matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = smoothed_files;

            matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
            matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
            matlabbatch{1}.spm.stats.factorial_design.masking.im = 0;
            matlabbatch{1}.spm.stats.factorial_design.masking.em = {mask_file};

            matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
            matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
            matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

            matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat_path};
            matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
            matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

            matlabbatch{3}.spm.stats.con.spmmat = {spm_mat_path};
            matlabbatch{3}.spm.stats.con.delete = 0;

            con_name = sprintf('RSA session-%s | %s | %s | %s > 0', collapsed_tag, input_tag, mode_tag, model_tag);
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = con_name;
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

            fprintf('Running SPM batch in: %s\n', output_dir);
            spm_jobman('run', matlabbatch);

            fprintf('  -> Done. spmT: %s\n', fullfile(output_dir, 'spmT_0001.nii'));
        end
    end
end

fprintf('\nAll done.\n');
