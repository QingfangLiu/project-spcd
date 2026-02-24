% aggregate_rsa_group_level_spm_probe_cuewise_NEWNAMING.m
%
% Group-level SPM analysis for RSA searchlight maps (probe_cuewise layout).
%
% INPUT layout:
%   <base_dir>/probe_cuewise/<session>/<input-tag>/
%       sub-01_mode-both_crossnobis_spearman_model-identity_subset-all_odor_mask-0_radius-8.0mm_whiten-1.nii.gz
%       ...
%
% OUTPUT layout:
%   <base_dir>/probe_cuewise/group_stats_spm/
%       session-<session>/<input>/<mode>/<model>/<metric>/<rsa>/<subset>/odor_mask-<0|1>/whiten-<0|1>/radius-<X>/
%       session-collapsed/<...same...>/
%
% Core logic:
%   - gunzip .nii.gz -> .nii
%   - load GM mask once
%   - set outside mask to NaN
%   - smooth
%   - SPM one-sample t-test
%   - one-sided contrast (> 0)
%   - skip if SPM.mat exists
%
% Requires SPM12 on MATLAB path.

%% ---------- Settings ----------
addpath('/Users/liuq13/Applications/spm12/');

base_dir  = '/Users/liuq13/SPCD_data/rsa_maps';
analysis_subfolder = 'probe_cuewise';

sessions = {'premeal', 'postmeal'};
inputs   = {'input-beta', 'input-tmap'};

modes    = {'both', 'img1only', 'img2only'};
models   = {'identity', 'value', 'association'}; % include if you generate it

neural_metrics = {'crossnobis', 'corr'};  % include only those you saved
rsa_methods    = {'spearman', 'kendall', 'regression'}; % include only those you saved

cue_subsets      = {'all', 'conditioned', 'preconditioned'};
odor_mask_flags  = [0, 1];
whiten_flags     = [0, 1];

radius_mm  = 8.0;               % must match file token radius-8.0mm
radius_tag = sprintf('radius-%.1fmm', radius_mm);
fwhm       = 6;                 % smoothing kernel in mm

group_stats_suffix = 'group_stats_spm';

mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';

skip_subs = {'sub-19', 'sub-33'};

collapsed_tag = 'collapsed';

%% ---------- Init SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

analysis_root = fullfile(base_dir, analysis_subfolder);

% load GM mask once
M = spm_read_vols(spm_vol(mask_file)) > 0;

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

                for nmi = 1:numel(neural_metrics)
                    metric_tag = neural_metrics{nmi};

                    for rsi = 1:numel(rsa_methods)
                        rsa_tag = rsa_methods{rsi};

                        for csi = 1:numel(cue_subsets)
                            subset_tag = cue_subsets{csi};

                            for omi = 1:numel(odor_mask_flags)
                                odor_mask_flag = odor_mask_flags(omi);

                                for wi = 1:numel(whiten_flags)
                                    whiten_flag = whiten_flags(wi);

                                    % Output dir
                                    output_dir = fullfile(analysis_root, group_stats_suffix, ...
                                        sprintf('session-%s', session_tag), input_tag, ...
                                        sprintf('mode-%s', mode_tag), ...
                                        sprintf('model-%s', model_tag), ...
                                        sprintf('metric-%s', metric_tag), ...
                                        sprintf('rsa-%s', rsa_tag), ...
                                        sprintf('subset-%s', subset_tag), ...
                                        sprintf('odor_mask-%d', odor_mask_flag), ...
                                        sprintf('whiten-%d', whiten_flag), ...
                                        radius_tag);
                                    ensure_dir(output_dir);

                                    spm_mat_path = fullfile(output_dir, 'SPM.mat');
                                    if exist(spm_mat_path, 'file')
                                        fprintf('SPM.mat exists, skipping: %s\n', output_dir);
                                        continue;
                                    end

                                    spm_input_dir = fullfile(output_dir, 'spm_inputs');
                                    ensure_dir(spm_input_dir);

                                    % New filename pattern (session is in folder, NOT in filename)
                                    % Example:
                                    % sub-01_mode-both_crossnobis_spearman_model-identity_subset-all_odor_mask-0_radius-8.0mm_whiten-1.nii.gz
                                    pat = sprintf('sub-*_mode-%s_%s_%s_model-%s_subset-%s_odor_mask-%d_%s_whiten-%d.nii.gz', ...
                                        mode_tag, metric_tag, rsa_tag, model_tag, subset_tag, odor_mask_flag, radius_tag, whiten_flag);

                                    all_gz = dir(fullfile(input_dir, pat));

                                    % Filter skip_subs (substring match)
                                    keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), all_gz);
                                    all_gz = all_gz(keep_idx);

                                    fprintf('\n>>> GROUP: session=%s | input=%s | mode=%s | model=%s | metric=%s | rsa=%s | subset=%s | odor_mask=%d | whiten=%d\n', ...
                                        session_tag, input_tag, mode_tag, model_tag, metric_tag, rsa_tag, subset_tag, odor_mask_flag, whiten_flag);
                                    fprintf('Found %d .nii.gz maps.\n', numel(all_gz));

                                    if isempty(all_gz)
                                        continue;
                                    end

                                    smoothed_files = cell(0,1);

                                    for fi = 1:numel(all_gz)
                                        gz_path = fullfile(all_gz(fi).folder, all_gz(fi).name);

                                        nii_path = ensure_nii_from_niigz(gz_path);

                                        V = spm_vol(nii_path);
                                        Y = spm_read_vols(V);

                                        [~, base_name, ~] = fileparts(nii_path); % strips ".nii"
                                        fprintf('  Mask->NaN + smooth: %s\n', base_name);

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

                                    con_name = sprintf('RSA %s | %s | %s | %s | %s | %s | subset-%s | odor_mask-%d | whiten-%d | %s > 0', ...
                                        session_tag, input_tag, mode_tag, model_tag, metric_tag, rsa_tag, subset_tag, odor_mask_flag, whiten_flag, radius_tag);

                                    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = con_name;
                                    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
                                    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

                                    fprintf('Running SPM batch in: %s\n', output_dir);
                                    spm_jobman('run', matlabbatch);

                                    fprintf('  -> Done. spmT: %s\n', fullfile(output_dir, 'spmT_0001.nii'));

                                end % whiten_flags
                            end % odor_mask_flags
                        end % cue_subsets
                    end % rsa_methods
                end % neural_metrics
            end % models
        end % modes
    end % inputs
end % sessions

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

            for nmi = 1:numel(neural_metrics)
                metric_tag = neural_metrics{nmi};

                for rsi = 1:numel(rsa_methods)
                    rsa_tag = rsa_methods{rsi};

                    for csi = 1:numel(cue_subsets)
                        subset_tag = cue_subsets{csi};

                        for omi = 1:numel(odor_mask_flags)
                            odor_mask_flag = odor_mask_flags(omi);

                            for wi = 1:numel(whiten_flags)
                                whiten_flag = whiten_flags(wi);

                                output_dir = fullfile(analysis_root, group_stats_suffix, ...
                                    sprintf('session-%s', collapsed_tag), input_tag, ...
                                    sprintf('mode-%s', mode_tag), ...
                                    sprintf('model-%s', model_tag), ...
                                    sprintf('metric-%s', metric_tag), ...
                                    sprintf('rsa-%s', rsa_tag), ...
                                    sprintf('subset-%s', subset_tag), ...
                                    sprintf('odor_mask-%d', odor_mask_flag), ...
                                    sprintf('whiten-%d', whiten_flag), ...
                                    radius_tag);
                                ensure_dir(output_dir);

                                spm_mat_path = fullfile(output_dir, 'SPM.mat');
                                if exist(spm_mat_path, 'file')
                                    fprintf('SPM.mat exists, skipping: %s\n', output_dir);
                                    continue;
                                end

                                spm_input_dir = fullfile(output_dir, 'spm_inputs');
                                ensure_dir(spm_input_dir);

                                pre_dir  = fullfile(analysis_root, 'premeal',  input_tag);
                                post_dir = fullfile(analysis_root, 'postmeal', input_tag);

                                if ~exist(pre_dir, 'dir') || ~exist(post_dir, 'dir')
                                    fprintf('Missing premeal/postmeal input dirs for %s, skipping.\n', input_tag);
                                    continue;
                                end

                                pre_pat = sprintf('sub-*_mode-%s_%s_%s_model-%s_subset-%s_odor_mask-%d_%s_whiten-%d.nii.gz', ...
                                    mode_tag, metric_tag, rsa_tag, model_tag, subset_tag, odor_mask_flag, radius_tag, whiten_flag);
                                post_pat = pre_pat;

                                pre_gz  = dir(fullfile(pre_dir,  pre_pat));
                                post_gz = dir(fullfile(post_dir, post_pat));

                                pre_keep  = ~arrayfun(@(f) any(contains(f.name, skip_subs)), pre_gz);
                                post_keep = ~arrayfun(@(f) any(contains(f.name, skip_subs)), post_gz);
                                pre_gz  = pre_gz(pre_keep);
                                post_gz = post_gz(post_keep);

                                fprintf('\n>>> COLLAPSED GROUP: input=%s | mode=%s | model=%s | metric=%s | rsa=%s | subset=%s | odor_mask=%d | whiten=%d\n', ...
                                    input_tag, mode_tag, model_tag, metric_tag, rsa_tag, subset_tag, odor_mask_flag, whiten_flag);
                                fprintf('Premeal files: %d | Postmeal files: %d\n', numel(pre_gz), numel(post_gz));

                                if isempty(pre_gz) || isempty(post_gz)
                                    fprintf('Not enough files to collapse; skipping.\n');
                                    continue;
                                end

                                pre_map  = containers.Map();
                                post_map = containers.Map();

                                for k = 1:numel(pre_gz)
                                    nm = pre_gz(k).name;
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

                                pre_keys = pre_map.keys;
                                subj_keys = {};
                                for k = 1:numel(pre_keys)
                                    s = pre_keys{k};
                                    if isKey(post_map, s)
                                        subj_keys{end+1} = s; %#ok<AGROW>
                                    end
                                end
                                subj_keys = sort(subj_keys);

                                fprintf('Subjects with both sessions: %d\n', numel(subj_keys));
                                if isempty(subj_keys)
                                    fprintf('No overlapping subjects; skipping.\n');
                                    continue;
                                end

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

                                    Ypre(~M)  = NaN;
                                    Ypost(~M) = NaN;

                                    Yavg = (Ypre + Ypost) ./ 2;

                                    base_name = sprintf('%s_session-%s_%s_mode-%s_%s_%s_%s_subset-%s_odor_mask-%d_whiten-%d_%s', ...
                                        sid, collapsed_tag, input_tag, mode_tag, metric_tag, rsa_tag, model_tag, subset_tag, odor_mask_flag, whiten_flag, radius_tag);

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

                                con_name = sprintf('RSA session-%s | %s | %s | %s | %s | %s | subset-%s | odor_mask-%d | whiten-%d | %s > 0', ...
                                    collapsed_tag, input_tag, mode_tag, model_tag, metric_tag, rsa_tag, subset_tag, odor_mask_flag, whiten_flag, radius_tag);

                                matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = con_name;
                                matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
                                matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

                                fprintf('Running SPM batch in: %s\n', output_dir);
                                spm_jobman('run', matlabbatch);

                                fprintf('  -> Done. spmT: %s\n', fullfile(output_dir, 'spmT_0001.nii'));

                            end % whiten_flags
                        end % odor_mask_flags
                    end % cue_subsets
                end % rsa_methods
            end % neural_metrics
        end % models
    end % modes
end % inputs

fprintf('\nAll done.\n');

%% ---------- Local helper functions (must be at end of script) ----------
function ensure_dir(d)
    if ~exist(d,'dir')
        mkdir(d);
    end
end

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

function smooth_file_with_vol(V, Y, M, spm_input_dir, base_name, fwhm)
    Y(~M) = NaN;

    masked_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
    Vc = V;
    Vc.fname = masked_path;
    spm_write_vol(Vc, Y);

    smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
    spm_smooth(masked_path, smooth_path, [fwhm fwhm fwhm]);
end