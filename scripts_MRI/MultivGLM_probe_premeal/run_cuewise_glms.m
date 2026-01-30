
% run_cuewise_glms.m
%
% - 12 cue regressors defined by cue_index (1..12)
% - separate decision + button regressors
% - nuisance via multi_reg
% - run-wise GLMs (runs 1..3; keep only those with events TSV)
% - 3 cue inclusion modes: img1only, img2only, both
%
% Output structure (Option B):
%   SPCD_data/GLM/<task>_cuewise_2mm/<subID>/<mode>/
%     run-01_spm/   (contains SPM.mat, ResMS.nii, spmT_*.nii, etc.)
%     tmap_task-<task>_run-01_cue-01.nii ... cue-12.nii
%     beta_task-<task>_run-01_cue-01.nii ... cue-12.nii
%     <subID>_task-<task>_mode-<mode>_run-01_index.csv
%
% This script generates t-maps via contrasts: cue_01..cue_12 > baseline.
%
% ========================================================================

clear; clc;

% ---- SPM PATH ----
addpath('/Users/liuq13/Applications/spm12/');
spm('Defaults','fMRI');
spm_jobman('initcfg');

% ---- USER SETTINGS ----
HomeDir  = '/Users/liuq13/SPCD_data';
niiRoot  = fullfile(HomeDir, 'nii');

taskname = 'premeal';      % 'premeal' or 'postmeal'
mode     = 'both';         % 'img1only' | 'img2only' | 'both'

TR = 1.5;

% output root
glmRoot = fullfile(HomeDir, 'GLM', sprintf('%s_cuewise_2mm', taskname));
if ~exist(glmRoot, 'dir'), mkdir(glmRoot); end

fprintf('=== Cue-wise GLM + t-maps (tidy) | task=%s | mode=%s ===\n', taskname, mode);

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

        % permanent run SPM folder
        run_spm_dir = fullfile(subj_out_dir, sprintf('run-%02d_spm', r));
        if ~exist(run_spm_dir, 'dir'), mkdir(run_spm_dir); end

        % if already estimated, skip to avoid overwriting
        spmmat_path = fullfile(run_spm_dir, 'SPM.mat');
        if exist(spmmat_path, 'file') == 2
            fprintf('    ↪ SPM.mat exists, skipping estimation: %s\n', spmmat_path);
            % Still ensure tmap copies exist (optional); comment out if undesired
            try
                ensure_tmap_copies(run_spm_dir, subj_out_dir, taskname, r);
            catch ME
                fprintf('    ⚠ Could not ensure t-map copies: %s\n', ME.message);
            end
            continue;
        end

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

        % ---- Build cue condition onsets/durations (12 conditions) ----
        cue_cond_on  = cell(12,1);
        cue_cond_dur = cell(12,1);
        cue_n_events = zeros(12,1);

        for c = 1:12
            m = (cue_idx_num == c);
            cue_cond_on{c}  = cue_on(m);
            cue_cond_dur{c} = cue_dur(m);
            cue_n_events(c) = sum(m);

            if cue_n_events(c) == 0
                fprintf('    ⚠  %s run %d: cue_index=%d has 0 events (mode=%s)\n', SubID, r, c, mode);
            end
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

        % 12 cue conditions
        for c = 1:12
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(c).name     = sprintf('cue_%02d', c);
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(c).onset    = cue_cond_on{c}';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(c).duration = cue_cond_dur{c}';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(c).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(c).pmod     = struct('name',{},'param',{},'poly',{});
        end

        % decision
        idx_dec = 13;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(idx_dec).name     = 'decision';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(idx_dec).onset    = dec_on';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(idx_dec).duration = dec_dur';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(idx_dec).tmod     = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(idx_dec).pmod     = struct('name',{},'param',{},'poly',{});

        % button (optional)
        if has_button
            idx_btn = 14;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(idx_btn).name     = 'button';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(idx_btn).onset    = btn_on';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(idx_btn).duration = zeros(size(btn_on'))';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(idx_btn).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(idx_btn).pmod     = struct('name',{},'param',{},'poly',{});
        end

        % nuisance + HPF
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {regfile};
        matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;

        % estimate
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(run_spm_dir, 'SPM.mat')};

        % Run spec+est
        spm_jobman('run', matlabbatch);

        % ---- Add contrasts (12 t-maps: cue_01..cue_12 > baseline) ----
        conBatch = {};
        conBatch{1}.spm.stats.con.spmmat = {fullfile(run_spm_dir, 'SPM.mat')};
        conBatch{1}.spm.stats.con.delete = 0;  % keep any existing

        % Determine number of regressors in this session design
        % (use SPM.xX.name after estimation to get correct vector length)
        S = load(fullfile(run_spm_dir, 'SPM.mat'));
        nReg = size(S.SPM.xX.X, 2);

        for c = 1:12
            w = zeros(1, nReg);
            w(c) = 1; % cue regressors are first 12 by construction
            conBatch{1}.spm.stats.con.consess{c}.tcon.name = sprintf('cue_%02d_gt_baseline', c);
            conBatch{1}.spm.stats.con.consess{c}.tcon.weights = w;
            conBatch{1}.spm.stats.con.consess{c}.tcon.sessrep = 'none';
        end

        spm_jobman('run', conBatch);

        % ---- Copy/rename cue betas and t-maps into subj_out_dir ----
        % We only want to "publish" 12 cue betas and 12 cue t-maps with clean names.
        idxTbl = table('Size',[12 10], ...
            'VariableTypes', {'string','string','string','string','double','double','double','double','string','string'}, ...
            'VariableNames', {'sub','task','mode','run_spm_dir','run','cue_index','n_events','n_vols','beta_file','tmap_file'});

        for c = 1:12
            % beta: beta_0001..beta_0012 correspond to cue_01..cue_12
            src_beta = fullfile(run_spm_dir, sprintf('beta_%04d.nii', c));
            if ~exist(src_beta, 'file')
                error('Missing expected cue beta: %s', src_beta);
            end
            beta_name = sprintf('beta_task-%s_run-%02d_cue-%02d.nii', taskname, r, c);
            dst_beta  = fullfile(subj_out_dir, beta_name);
            copyfile(src_beta, dst_beta);

            % t-map: by default, SPM creates spmT_0001..spmT_0012 in contrast order
            src_t = fullfile(run_spm_dir, sprintf('spmT_%04d.nii', c));
            if ~exist(src_t, 'file')
                error('Missing expected cue t-map: %s', src_t);
            end
            t_name  = sprintf('tmap_task-%s_run-%02d_cue-%02d.nii', taskname, r, c);
            dst_t   = fullfile(subj_out_dir, t_name);
            copyfile(src_t, dst_t);

            idxTbl.sub(c)        = string(SubID);
            idxTbl.task(c)       = string(taskname);
            idxTbl.mode(c)       = string(mode);
            idxTbl.run_spm_dir(c)= string(run_spm_dir);
            idxTbl.run(c)        = r;
            idxTbl.cue_index(c)  = c;
            idxTbl.n_events(c)   = cue_n_events(c);
            idxTbl.n_vols(c)     = n_vols;
            idxTbl.beta_file(c)  = string(beta_name);
            idxTbl.tmap_file(c)  = string(t_name);
        end

        out_csv = fullfile(subj_out_dir, sprintf('%s_task-%s_mode-%s_run-%02d_index.csv', ...
            SubID, taskname, mode, r));
        writetable(idxTbl, out_csv);
        fprintf('    ✅ Saved betas+t-maps+index: %s\n', out_csv);

    end
end

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
% Helper: if estimation was skipped because SPM.mat exists, ensure tmap copies exist
% ========================================================================
function ensure_tmap_copies(run_spm_dir, subj_out_dir, taskname, r)
    for c = 1:12
        src_t = fullfile(run_spm_dir, sprintf('spmT_%04d.nii', c));
        if exist(src_t, 'file') ~= 2
            error('Missing %s; contrasts may not have been run.', src_t);
        end
        dst_t = fullfile(subj_out_dir, sprintf('tmap_task-%s_run-%02d_cue-%02d.nii', taskname, r, c));
        if exist(dst_t, 'file') ~= 2
            copyfile(src_t, dst_t);
        end
    end
end
