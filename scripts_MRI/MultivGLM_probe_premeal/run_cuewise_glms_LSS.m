
% run_cuewise_glms_LSS.m
%
% LSS-style cue GLMs (one GLM per cue per run):
%   - Regressor 1: cue_<c>_target (only cue c)
%   - Regressor 2: cue_<c>_other  (all other cues lumped)
%   - Regressor 3: decision
%   - Regressor 4 (optional): button
%   - nuisance via multi_reg
%
% Output root is 
%   SPCD_data/GLM/<task>_cuewiseLSS_2mm/<subID>/<mode>/
%
% Tidy outputs saved to subj_out_dir:
%   beta_task-<task>_run-XX_cue-YY.nii
%   tmap_task-<task>_run-XX_cue-YY.nii
%   <subID>_task-<task>_mode-<mode>_run-XX_index.csv
%
% SPM cleanup:
%   After copying tidy beta/tmap, delete all SPM-generated files EXCEPT SPM.mat.
%
% Skip logic:
%   Skip a cue GLM if BOTH tidy beta and tidy tmap already exist.

clear; clc;

% ---- SPM PATH ----
addpath('/Users/liuq13/Applications/spm12/');
spm('Defaults','fMRI');
spm_jobman('initcfg');

% ---- USER SETTINGS ----
HomeDir  = '/Users/liuq13/SPCD_data';
niiRoot  = fullfile(HomeDir, 'nii');

taskname = 'premeal';      % 'premeal' or 'postmeal'
mode     = 'both';     % 'img1only' | 'img2only' | 'both'
TR = 1.5;

% output root (parallel to "<task>_cuewise_2mm")
glmRoot = fullfile(HomeDir, 'GLM', sprintf('%s_cuewiseLSS_2mm', taskname));
if ~exist(glmRoot, 'dir'), mkdir(glmRoot); end

fprintf('=== Cue-wise LSS GLM + tidy betas/t-maps | task=%s | mode=%s ===\n', taskname, mode);
fprintf('=== Output root: %s ===\n', glmRoot);

% ---- SUBJECTS ----
subs = dir(fullfile(niiRoot, 'sub-*'));
subs = subs([subs.isdir]);

for s = 1:numel(subs)
    SubID = subs(s).name;
    fprintf('\n=== %s ===\n', SubID);

    % mode-specific output dir
    subj_out_dir = fullfile(glmRoot, SubID, mode);
    if ~exist(subj_out_dir, 'dir'), mkdir(subj_out_dir); end

    % events dir
    ev_dir = fullfile(niiRoot, SubID, 'func_PAID', taskname);

    % runs 1..3, keep only those that have events TSV
    run_list = 1:3;
    has_ev = arrayfun(@(r) exist(fullfile(ev_dir, ...
        sprintf('%s_task-%s_run-%d_events.tsv', SubID, taskname, r)), 'file')==2, run_list);
    run_list = run_list(has_ev);

    if isempty(run_list)
        fprintf('  (no events files found; skipping)\n');
        continue;
    end

    for r = run_list(:)'

        % hard skip: sub-36 run 3
        if strcmp(SubID,'sub-36') && r==3
            fprintf('  ⚠  Skipping %s run %d (known bad MRI)\n', SubID, r);
            continue
        end

        fprintf('  -- Run %d --\n', r);

        % ---- Load events TSV ----
        tsv_file = fullfile(ev_dir, sprintf('%s_task-%s_run-%d_events.tsv', SubID, taskname, r));
        if ~exist(tsv_file,'file')
            fprintf('  ❌ Missing events: %s\n', tsv_file);
            continue
        end
        E = readtable(tsv_file, 'FileType','text', 'Delimiter','\t', 'TextType','string');

        % cue inclusion based on mode
        cue_types = get_cue_types_from_mode(mode);
        is_cue = ismember(E.trial_type, cue_types);

        is_dec = (E.trial_type == "decision");
        is_btn = (E.trial_type == "button");

        cue_on  = E.onset(is_cue);
        cue_dur = E.duration(is_cue);      % expected 3
        cue_idx = E.cue_index(is_cue);

        if isempty(cue_on)
            error('No cue events found for %s run %d with mode=%s.', SubID, r, mode);
        end

        % guardrail: cue_index must be numeric in 1..12
        cue_idx_num = cue_idx;
        if iscell(cue_idx_num) || isstring(cue_idx_num)
            cue_idx_num = str2double(string(cue_idx_num));
        end
        if any(isnan(cue_idx_num)) || any(~ismember(cue_idx_num, 1:12))
            offending_rows = find(is_cue);
            error(['Invalid/missing cue_index detected for %s run %d (mode=%s). ' ...
                   'Example offending TSV row index: %d'], ...
                   SubID, r, mode, offending_rows(1));
        end

        % decision regressor
        dec_on  = E.onset(is_dec);
        dec_dur = E.duration(is_dec);      % expected 2

        % button regressor
        btn_on = E.onset(is_btn);          % may be empty
        has_button = ~isempty(btn_on);

        % ---- Locate s2w2 4D NIfTI ----
        scandir = fullfile(niiRoot, SubID, 'func_PAID', taskname);
        nii_struct = dir(fullfile(scandir, sprintf('s2w2Scan2_%s_run_%d_*_PAID.nii', taskname, r)));
        if isempty(nii_struct)
            fprintf('  ❌ No s2w2 .nii for run %d in %s\n', r, scandir);
            continue
        end
        nii_file = fullfile(scandir, nii_struct(1).name);
        vols = cellstr(spm_select('expand', nii_file));
        n_vols = numel(vols);

        % ---- Nuisance regressors ----
        regfile = fullfile(HomeDir, 'NRegressor', SubID, ...
            sprintf('nuisance_regressors_%s_run_%d_zscored.txt', taskname, r));
        if ~exist(regfile, 'file')
            fprintf('  ❌ Missing nuisance: %s\n', regfile);
            continue
        end

        % Index table (12 rows, one per cue GLM)
        idxTbl = table('Size',[12 12], ...
            'VariableTypes', {'string','string','string','string','double','double','double','double','double','double','string','string'}, ...
            'VariableNames', {'sub','task','mode','run_spm_dir','run','cue_index','n_events_target','n_events_other','n_vols','nReg','beta_file','tmap_file'});

        for c = 1:12
            beta_name = sprintf('beta_task-%s_run-%02d_cue-%02d.nii', taskname, r, c);
            t_name    = sprintf('tmap_task-%s_run-%02d_cue-%02d.nii', taskname, r, c);

            dst_beta  = fullfile(subj_out_dir, beta_name);
            dst_t     = fullfile(subj_out_dir, t_name);

            % per-cue SPM folder
            run_spm_dir = fullfile(subj_out_dir, sprintf('run-%02d_cue-%02d_spm', r, c));
            if ~exist(run_spm_dir, 'dir'), mkdir(run_spm_dir); end
            spmmat_path = fullfile(run_spm_dir, 'SPM.mat');

            % Skip if tidy outputs already exist
            if exist(dst_beta, 'file') == 2 && exist(dst_t, 'file') == 2
                fprintf('    [skip] cue %02d (tidy beta+tmap already exist)\n', c);

                % populate index row (counts can still be computed)
                n_target = sum(cue_idx_num == c);
                n_other  = sum(cue_idx_num ~= c);

                % nReg: try to read from SPM.mat if it exists
                nReg = NaN;
                if exist(spmmat_path, 'file') == 2
                    try
                        S = load(spmmat_path);
                        nReg = size(S.SPM.xX.X, 2);
                    catch
                        nReg = NaN;
                    end
                end

                idxTbl.sub(c)            = string(SubID);
                idxTbl.task(c)           = string(taskname);
                idxTbl.mode(c)           = string(mode);
                idxTbl.run_spm_dir(c)    = string(run_spm_dir);
                idxTbl.run(c)            = r;
                idxTbl.cue_index(c)      = c;
                idxTbl.n_events_target(c)= n_target;
                idxTbl.n_events_other(c) = n_other;
                idxTbl.n_vols(c)         = n_vols;
                idxTbl.nReg(c)           = nReg;
                idxTbl.beta_file(c)      = string(beta_name);
                idxTbl.tmap_file(c)      = string(t_name);
                continue;
            end

            fprintf('    [LSS] cue %02d\n', c);

            % If tidy outputs are missing but SPM.mat exists, the required beta_0001/spmT_0001
            % were likely deleted previously. Re-run from scratch by removing old SPM.mat.
            if exist(spmmat_path, 'file') == 2
                fprintf('      ⚠  Tidy outputs missing but SPM.mat exists; re-estimating (deleting old SPM.mat)\n');
                delete(spmmat_path);
            end

            % Target vs Other cue onsets/durations
            m_target = (cue_idx_num == c);
            m_other  = (cue_idx_num ~= c);

            target_on  = cue_on(m_target);
            target_dur = cue_dur(m_target);

            other_on   = cue_on(m_other);
            other_dur  = cue_dur(m_other);

            n_target = sum(m_target);
            n_other  = sum(m_other);

            if n_target == 0
                fprintf('      ⚠  %s run %d: cue_index=%d has 0 events (mode=%s)\n', SubID, r, c, mode);
            end

            % ---- Build SPM job: spec + estimate ----
            matlabbatch = {};

            % Spec
            matlabbatch{1}.spm.stats.fmri_spec.dir = {run_spm_dir};
            matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
            matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t  = 16;
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
            matlabbatch{1}.spm.stats.fmri_spec.sess.scans = vols;

            % mimic old approach
            matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.1;
            matlabbatch{1}.spm.stats.fmri_spec.global = 'None';

            % Cond 1: target cue
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name     = sprintf('cue_%02d_target', c);
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset    = target_on';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = target_dur';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod     = struct('name',{},'param',{},'poly',{});

            % Cond 2: other cues lumped
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name     = sprintf('cue_%02d_other', c);
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset    = other_on';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = other_dur';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod     = struct('name',{},'param',{},'poly',{});

            % decision
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).name     = 'decision';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).onset    = dec_on';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).duration = dec_dur';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).pmod     = struct('name',{},'param',{},'poly',{});

            % button (optional)
            if has_button
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).name     = 'button';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).onset    = btn_on';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).duration = zeros(size(btn_on'))';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).tmod     = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).pmod     = struct('name',{},'param',{},'poly',{});
            end

            % nuisance + HPF
            matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {regfile};
            matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;

            % estimate
            matlabbatch{2}.spm.stats.fmri_est.spmmat = {spmmat_path};

            % Run spec+est
            spm_jobman('run', matlabbatch);

            % ---- Add contrast: target > baseline ----
            conBatch = {};
            conBatch{1}.spm.stats.con.spmmat = {spmmat_path};
            conBatch{1}.spm.stats.con.delete = 0;

            S = load(spmmat_path);
            nReg = size(S.SPM.xX.X, 2);

            w = zeros(1, nReg);
            w(1) = 1; % target cue is cond(1) by construction

            conBatch{1}.spm.stats.con.consess{1}.tcon.name = sprintf('cue_%02d_target_gt_baseline', c);
            conBatch{1}.spm.stats.con.consess{1}.tcon.weights = w;
            conBatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

            spm_jobman('run', conBatch);

            % ---- Copy out target beta + target tmap with tidy names ----
            src_beta = fullfile(run_spm_dir, 'beta_0001.nii'); % target always beta_0001
            if ~exist(src_beta, 'file')
                error('Missing expected target beta: %s', src_beta);
            end
            copyfile(src_beta, dst_beta);

            src_t = fullfile(run_spm_dir, 'spmT_0001.nii'); % only one contrast
            if ~exist(src_t, 'file')
                error('Missing expected target t-map: %s', src_t);
            end
            copyfile(src_t, dst_t);

            % ---- Cleanup SPM outputs but keep SPM.mat ----
            cleanup_spm_outputs_keep_spmmat(run_spm_dir);

            % ---- Index row ----
            idxTbl.sub(c)            = string(SubID);
            idxTbl.task(c)           = string(taskname);
            idxTbl.mode(c)           = string(mode);
            idxTbl.run_spm_dir(c)    = string(run_spm_dir);
            idxTbl.run(c)            = r;
            idxTbl.cue_index(c)      = c;
            idxTbl.n_events_target(c)= n_target;
            idxTbl.n_events_other(c) = n_other;
            idxTbl.n_vols(c)         = n_vols;
            idxTbl.nReg(c)           = nReg;
            idxTbl.beta_file(c)      = string(beta_name);
            idxTbl.tmap_file(c)      = string(t_name);

        end % cue loop

        out_csv = fullfile(subj_out_dir, sprintf('%s_task-%s_mode-%s_run-%02d_index.csv', ...
            SubID, taskname, mode, r));
        writetable(idxTbl, out_csv);
        fprintf('    ✅ Saved index: %s\n', out_csv);

    end % run loop
end % subject loop

fprintf('\nDone.\n');

% ========================================================================
% Helper: mode -> included cue trial_type list
% ========================================================================
function cue_types = get_cue_types_from_mode(mode)
    m = lower(string(mode));
    if m == "img1only"
        cue_types = ["cue_img1"];
    elseif m == "img2only"
        cue_types = ["cue_img2"];
    elseif m == "both"
        cue_types = ["cue_img1","cue_img2"];
    else
        error('Unknown mode: %s (expected img1only | img2only | both)', mode);
    end
end

% ========================================================================
% Helper: delete SPM outputs but keep SPM.mat
% ========================================================================
function cleanup_spm_outputs_keep_spmmat(run_spm_dir)
% Deletes all files in run_spm_dir except SPM.mat.
% Keeps folder and SPM.mat for provenance/debugging.

    if exist(run_spm_dir, 'dir') ~= 7
        return;
    end

    % Safety: only operate on folders that look like SPM output folders
    [~, folderName] = fileparts(run_spm_dir);
    if ~contains(folderName, "_spm")
        error('Refusing to clean non-SPM folder: %s', run_spm_dir);
    end

    % Delete everything except SPM.mat
    dd = dir(run_spm_dir);
    for k = 1:numel(dd)
        name = dd(k).name;
        if dd(k).isdir
            if strcmp(name,'.') || strcmp(name,'..')
                continue;
            end
            % SPM output folders rarely contain subfolders; remove if present
            rmdir(fullfile(run_spm_dir, name), 's');
        else
            if strcmp(name, 'SPM.mat')
                continue;
            end
            delete(fullfile(run_spm_dir, name));
        end
    end
end
