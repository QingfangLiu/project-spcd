%% ========================================================================
%  Odor Prediction (odorpred) — LSS single-trial GLM for CUE (EARLY 2 s)
%
%  For each CUE event (cue_stim) in a run, fit a mini-GLM with:
%    1) target cue_k            (2 s; t -> t+2)
%    2) all other cues pooled   (2 s; t -> t+2)
%    3) prediction_all          (3 s; t+2 -> t+5; all trials)
%    4) delay_all               (2 s; t+5 -> t+7; responded only)
%    5) odor_all                (2 s; t+7 -> t+9; responded only)
%    6) fixation_miss           (4 s; t+5 -> t+9; missed only)
%    7) button_all              (0 s sticks; at t+2+RT; responded only)
%
%  Inputs:
%    - NIfTI:
%        SPCD_data/nii/<subID>/func_PAID/odorpred/
%          s2w2Scan2_odorpred_run_<r>_*_PAID.nii
%    - Events:
%        SPCD_data/nii/<subID>/func_PAID/odorpred/
%          <subID>_task-odorpred_run-<r>_events.tsv
%        (expects columns at least:
%           onset, duration, trial_type, run, trial, cue_idx, odor_num,
%           response_label, rt, acc)
%    - Nuisance (z-scored):
%        SPCD_data/NRegressor/<subID>/
%          nuisance_regressors_odorpred_run_<r>_zscored.txt
%
%  Outputs:
%    - Betas moved to:
%        SPCD_data/GLM/odorpred_cueEARLY_trialwise_2mm/<subID>/
%          beta_run<r>_trial<trial>.nii
%    - Beta index CSV (per run):
%        .../<subID>_odorpred_run-<r>_beta_index.csv
%
%  Notes:
%    - Uses SPM12; canonical HRF (no derivatives), HPF=128 s, AR(1)
%    - Include ALL cue_stim trials as targets (responded + missed)
%    - Orthogonalization OFF (no pmods here); NO parametric modulators
%% ========================================================================

clear; clc;
addpath('/Users/liuq13/Applications/spm12/');

% ---- PATHS ----
HomeDir = '/Users/liuq13/SPCD_data';
niiRoot = fullfile(HomeDir, 'nii');
glmRoot = fullfile(HomeDir, 'GLM', 'odorpred_cueEARLY_trialwise_2mm');
taskname = 'odorpred';
if ~exist(glmRoot, 'dir'), mkdir(glmRoot); end

TR = 1.5;

% ---- SUBJECTS ----
subs = dir(fullfile(niiRoot, 'sub-*')); subs = subs([subs.isdir]);

for s = 1:numel(subs)
    SubID = subs(s).name;
    fprintf('\n=== LSS-CUE-EARLY | %s ===\n', SubID);

    % where betas will live (flat folder per subject)
    subj_beta_dir = fullfile(glmRoot, SubID);
    if ~exist(subj_beta_dir, 'dir'), mkdir(subj_beta_dir); end

    % Runs that actually have events
    run_list = 1:3;
    ev_dir = fullfile(niiRoot, SubID, 'func_PAID', taskname);
    has_ev = arrayfun(@(r) exist(fullfile(ev_dir, ...
        sprintf('%s_task-%s_run-%d_events.tsv', SubID, taskname, r)), 'file')==2, run_list);
    run_list = run_list(has_ev);

    for r = run_list(:)'
        fprintf('  -- Run %d --\n', r);

        % ---- Load events ----
        tsv_file = fullfile(ev_dir, sprintf('%s_task-%s_run-%d_events.tsv', SubID, taskname, r));
        E = readtable(tsv_file, 'FileType','text','Delimiter','\t', 'VariableNamingRule','preserve');

        % Select event sets we need
        is_cue  = strcmp(E.trial_type,'cue_stim');             % include ALL
        is_pred = strcmp(E.trial_type,'prediction');           % all trials
        is_del  = strcmp(E.trial_type,'delay');                % responded only
        is_odor = strcmp(E.trial_type,'odor');                 % responded only
        is_fix  = strcmp(E.trial_type,'fixation');             % missed only (replaces delay+odor)
        is_btn  = strcmp(E.trial_type,'button');               % responded only

        cue_on   = E.onset(is_cue);
        cue_durE = repmat(2, sum(is_cue), 1);  % EARLY 2 s window regardless of actual duration
        cue_trial= E.trial(is_cue);
        cue_idx  = E.cue_idx(is_cue);
        % odor_num may be numeric or char—normalize to double for logging
        if ismember('odor_num', E.Properties.VariableNames)
            if isnumeric(E.odor_num)
                odor_num = E.odor_num(is_cue);
            else
                odor_num = str2double(string(E.odor_num(is_cue)));
            end
        else
            odor_num = nan(size(cue_on));
        end

        % From prediction rows, grab behavior by trial (for beta index)
        P = E(is_pred, {'trial','response_label','rt','acc'});
        P.response_label = string(P.response_label);
        
        pred_on  = E.onset(is_pred);
        pred_dur = E.duration(is_pred);    % 3s

        del_on   = E.onset(is_del);        % may be empty
        del_dur  = E.duration(is_del);     % 2s

        odor_on  = E.onset(is_odor);       % may be empty
        odor_dur = E.duration(is_odor);    % 2s

        fix_on   = E.onset(is_fix);        % missed trials only
        fix_dur  = E.duration(is_fix);     % 4s

        btn_on   = E.onset(is_btn);        % may be empty

        n_cue = numel(cue_on);
        if n_cue == 0
            fprintf('     ❌ No cue_stim events in %s run %d\n', SubID, r);
            continue
        end
        fprintf('     cues=%d, predictions=%d, delays=%d, odors=%d, fix=%d, buttons=%d\n', ...
                n_cue, numel(pred_on), numel(del_on), numel(odor_on), numel(fix_on), numel(btn_on));

        % ---- Locate s2w2 4D NIfTI ----
        scandir = fullfile(niiRoot, SubID, 'func_PAID', taskname);
        nii_struct = dir(fullfile(scandir, sprintf('s2w2Scan2_%s_run_%d_*_PAID.nii', taskname, r)));
        if isempty(nii_struct)
            fprintf('     ❌ No s2w2 .nii for run %d in %s\n', r, scandir);
            continue
        end
        nii_file = fullfile(scandir, nii_struct(1).name);
        vols = cellstr(spm_select('expand', nii_file));

        % ---- Nuisance regressors (z-scored) ----
        regfile = fullfile(HomeDir, 'NRegressor', SubID, ...
            sprintf('nuisance_regressors_%s_run_%d_zscored.txt', taskname, r));
        if ~exist(regfile, 'file')
            fprintf('     ❌ Missing nuisance: %s\n', regfile);
            continue
        end

        % ---- Beta index table for this run ----
        beta_log = table('Size',[0 10], ...
            'VariableTypes', {'string','double','double','double','string','double','double','string','double','double'}, ...
            'VariableNames', {'beta_file','run','trial','onset','trial_type','cue_idx','odor_num','response_label','rt','acc'});

        % ---- Loop over cue events (LSS mini-GLMs) ----
        for k = 1:n_cue
            target_on   = cue_on(k);      % 2 s early window
            target_dur  = cue_durE(k);
            other_mask  = true(n_cue,1); other_mask(k) = false;

            other_on    = cue_on(other_mask);
            other_dur   = cue_durE(other_mask);

            outdir = fullfile(glmRoot, 'work', SubID, sprintf('run%02d_cue%03d', r, k));
            if ~exist(outdir,'dir'), mkdir(outdir); end

            % -- Build SPM job --
            spm_jobman('initcfg');
            matlabbatch = {};

            % Spec
            matlabbatch{1}.spm.stats.fmri_spec.dir = {outdir};
            matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
            matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t  = 16;
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
            matlabbatch{1}.spm.stats.fmri_spec.sess.scans = vols;
            matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
            matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.1;
            matlabbatch{1}.spm.stats.fmri_spec.global = 'None';

            % 1) target cue_k (EARLY 2s)
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name     = 'cue_target_2s';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset    = target_on;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = target_dur;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod     = struct('name',{},'param',{},'poly',{});

            % 2) cue_others pooled (EARLY 2s)
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name     = 'cue_others_2s';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset    = other_on';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = other_dur';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod     = struct('name',{},'param',{},'poly',{});

            % 3) prediction_all
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).name     = 'prediction_all';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).onset    = pred_on';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).duration = pred_dur';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).pmod     = struct('name',{},'param',{},'poly',{});

            % 4) delay_all (pooled; may be empty)
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).name     = 'delay_all';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).onset    = del_on';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).duration = del_dur';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).pmod     = struct('name',{},'param',{},'poly',{});

            % 5) odor_all (responded only)
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).name     = 'odor_all';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).onset    = odor_on';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).duration = odor_dur';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).pmod     = struct('name',{},'param',{},'poly',{});

            % 6) fixation_miss (missed only)
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).name     = 'fixation_miss';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).onset    = fix_on';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).duration = fix_dur';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).pmod     = struct('name',{},'param',{},'poly',{});

            % 7) button_all sticks (only if present)
            if ~isempty(btn_on)
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(7).name     = 'button_all';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(7).onset    = btn_on';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(7).duration = zeros(size(btn_on'))';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(7).tmod     = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(7).pmod     = struct('name',{},'param',{},'poly',{});
            end

            % Nuisance + HPF
            matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {regfile};
            matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;

            % Estimation
            matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(outdir, 'SPM.mat')};

            % Run
            spm_jobman('run', matlabbatch);

            % Move beta_0001 to subject folder with trial-numbered name
            src_beta = fullfile(outdir, 'beta_0001.nii');
            trial_id = cue_trial(k);
            beta_name = sprintf('beta_run%d_trial%03d.nii', r, trial_id);
            dst_beta = fullfile(subj_beta_dir, beta_name);
            movefile(src_beta, dst_beta);

            % Metadata for index row (pull response info for this trial)
            rowP = P(P.trial == trial_id, :);
            if ~isempty(rowP)
                resp_lab = char(string(rowP.response_label(1)));
                rt_val   = rowP.rt(1);
                acc_val  = rowP.acc(1);
            else
                resp_lab = 'n/a'; rt_val = NaN; acc_val = NaN;
            end

            % Log mapping
            beta_log = [beta_log; { ...
                beta_name, ...           % beta_file
                r, ...                   % run
                trial_id, ...            % trial
                target_on, ...           % onset (cue)
                'cue', ...               % trial_type
                cue_idx(k), ...          % cue_idx
                odor_num(k), ...         % odor_num (if present)
                string(resp_lab), ...    % response_label
                rt_val, ...              % rt
                acc_val ...              % acc
            }];

            % Clean up mini-GLM folder
            delete(fullfile(outdir, '*'));
            rmdir(outdir);
        end

        % Save beta index CSV for this subject/run
        out_csv = fullfile(subj_beta_dir, sprintf('%s_%s_run-%d_beta_index.csv', SubID, taskname, r));
        writetable(beta_log, out_csv);
        fprintf('     ✅ Saved beta index: %s (rows=%d)\n', out_csv, height(beta_log));
    end
end
