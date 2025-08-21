
%% ========================================================================
%  Odor Prediction (odorpred) Event File Generator
%
%  Generates BIDS-like events.tsv files for the Odor Prediction task
%  for each subject in SPCD_data/nii/sub-*.
%
%  Input (per subject):
%    project-SPCD/beh_processed_subs/<subID>/odor_pred_dat.csv
%    Required columns (case-sensitive):
%      Run, trial, cue_idx, odor_num, ITIDur, resp, rt, acc
%
%  Output (per run):
%    SPCD_data/nii/<subID>/func_PAID/odorpred/<subID>_task-odorpred_run-<r>_events.tsv
%
%  Trial logic (absolute onsets in seconds):
%    - Run starts at t = 10 s (pre-run blank)
%    - cue_stim   : visual cue on screen
%         * duration = 7.0 s if responded (cue + prediction + delay)
%         * duration = 5.0 s if missed (cue + prediction only)
%         * onset = trial start (t)
%    - prediction : 3.0 s at t + 2.0 (always)
%    - button     : stick at t + 2.0 + rt (only if response present)
%    - delay      : 2.0 s at t + 5.0 (only if responded)
%    - odor       : 2.0 s at t + 7.0 (only if responded)
%    - fixation   : 4.0 s at t + 5.0 (only if missed; replaces delay+odor)
%
%  Clock advance (no ITI rows written):
%    t <- t + 9.0 + ITIDur   % active block is always 9 s
%
%  Response mapping:
%    resp = 0 -> 'left', 1 -> 'middle', 2 -> 'right', NaN -> missed
%
%  Columns written:
%    onset  duration  trial_type  run  trial  cue_idx  odor_num  response_label  rt  acc
%
%  Prints per-run stats and estimated run length (TR = 1.5 s).
%% ========================================================================

clear; clc;

% ---------- USER CONFIG ----------
HomeDir = '/Users/liuq13/SPCD_data';
niiDir  = fullfile(HomeDir, 'nii');
behDir  = '/Users/liuq13/project-SPCD/beh_processed_subs';

subs = dir(fullfile(niiDir,'sub-*')); subs = subs([subs.isdir]);

% Timing constants (s)
BLANK     = 10;   % pre-run blank
DUR_CUE   = 2;    % cue period before prediction (cue visible)
DUR_PRED  = 3;    % prediction window
DUR_DELAY = 2;    % delay (only if responded)
DUR_ODOR  = 2;    % odor delivery (only if responded)
ACTIVE_9  = 9;    % 2 + 3 + 2 + 2
TR        = 1.5;  % reporting only

for s = 1:numel(subs)
    SubID  = subs(s).name;
    in_csv = fullfile(behDir, SubID, 'odor_pred_dat.csv');
    outdir = fullfile(niiDir, SubID, 'func_PAID', 'odorpred');
    if ~exist(outdir,'dir'), mkdir(outdir); end

    if ~exist(in_csv,'file')
        fprintf('❌ Missing CSV for %s: %s\n', SubID, in_csv);
        continue
    end

    T = readtable(in_csv, 'VariableNamingRule','preserve');

    % Required columns
    reqCols = {'Run','trial','cue_idx','odor_num','ITIDur','resp','rt','acc'};
    miss = setdiff(reqCols, T.Properties.VariableNames);
    if ~isempty(miss)
        fprintf('❌ %s missing columns: %s\n', SubID, strjoin(miss, ', '));
        continue
    end

    run_list = unique(T.Run(~isnan(T.Run)))';
    fprintf('\n=== %s | odorpred events | runs: %s ===\n', SubID, sprintf('%g ', run_list));

    for r = run_list
        R = sortrows(T(T.Run==r,:), 'trial');
        t = BLANK;

        % Collectors
        onsets = []; durations = []; run_col = []; trial_col = [];
        trial_type = {};
        cue_idx_col  = [];
        odor_num_col = {};       % string to allow 'n/a'
        response_label = {};     % string
        rt_col  = [];
        acc_col = [];

        n_pred = 0; n_button = 0;

        for i = 1:height(R)
            tr   = R.trial(i);
            cuei = R.cue_idx(i);
            odor = R.odor_num(i);
            iti  = R.ITIDur(i);
            resp = R.resp(i);
            rt_i = R.rt(i);
            acci = R.acc(i);

            % Response mapping
            has_resp = ~isnan(resp);
            if has_resp
                if     resp==0, resp_lab = 'left';
                elseif resp==1, resp_lab = 'middle';
                elseif resp==2, resp_lab = 'right';
                else,            resp_lab = 'n/a';
                end
            else
                resp_lab = 'n/a';
            end

            % ---- cue_stim (visual presence of cue) ----
            if has_resp
                cue_dur = DUR_CUE + DUR_PRED + DUR_DELAY;   % 7 s
            else
                cue_dur = DUR_CUE + DUR_PRED;               % 5 s
            end
            onsets(end+1,1)      = t;
            durations(end+1,1)   = cue_dur;
            trial_type{end+1,1}  = 'cue_stim';
            run_col(end+1,1)     = r;
            trial_col(end+1,1)   = tr;
            cue_idx_col(end+1,1) = cuei;
            odor_num_col{end+1,1}= num2str(odor);   % carry label on cue
            response_label{end+1,1} = 'n/a';
            rt_col(end+1,1)      = NaN;
            acc_col(end+1,1)     = NaN;

            % ---- prediction (always) ----
            pred_on = t + DUR_CUE;
            onsets(end+1,1)      = pred_on;
            durations(end+1,1)   = DUR_PRED;
            trial_type{end+1,1}  = 'prediction';
            run_col(end+1,1)     = r;
            trial_col(end+1,1)   = tr;
            cue_idx_col(end+1,1) = cuei;
            odor_num_col{end+1,1}= 'n/a';
            response_label{end+1,1} = resp_lab;
            rt_col(end+1,1)      = has_resp * rt_i + (~has_resp) * NaN;
            acc_col(end+1,1)     = has_resp * acci + (~has_resp) * NaN;
            n_pred = n_pred + 1;

            if has_resp
                % ---- button (stick) ----
                btn_on = pred_on + rt_i;
                onsets(end+1,1)      = btn_on;
                durations(end+1,1)   = 0;
                trial_type{end+1,1}  = 'button';
                run_col(end+1,1)     = r;
                trial_col(end+1,1)   = tr;
                cue_idx_col(end+1,1) = cuei;
                odor_num_col{end+1,1}= 'n/a';
                response_label{end+1,1} = resp_lab;
                rt_col(end+1,1)      = rt_i;
                acc_col(end+1,1)     = acci;
                n_button = n_button + 1;

                % ---- delay ----
                del_on = t + DUR_CUE + DUR_PRED;     % = t + 5
                onsets(end+1,1)      = del_on;
                durations(end+1,1)   = DUR_DELAY;
                trial_type{end+1,1}  = 'delay';
                run_col(end+1,1)     = r;
                trial_col(end+1,1)   = tr;
                cue_idx_col(end+1,1) = cuei;
                odor_num_col{end+1,1}= 'n/a';
                response_label{end+1,1} = 'n/a';
                rt_col(end+1,1)      = NaN;
                acc_col(end+1,1)     = NaN;

                % ---- odor ----
                odor_on = del_on + DUR_DELAY;         % = t + 7
                onsets(end+1,1)      = odor_on;
                durations(end+1,1)   = DUR_ODOR;
                trial_type{end+1,1}  = 'odor';
                run_col(end+1,1)     = r;
                trial_col(end+1,1)   = tr;
                cue_idx_col(end+1,1) = cuei;
                odor_num_col{end+1,1}= num2str(odor); % carry label on odor
                response_label{end+1,1} = 'n/a';
                rt_col(end+1,1)      = NaN;
                acc_col(end+1,1)     = NaN;

            else
                % ---- fixation (replaces delay + odor) ----
                fix_on = t + DUR_CUE + DUR_PRED;      % = t + 5
                onsets(end+1,1)      = fix_on;
                durations(end+1,1)   = DUR_DELAY + DUR_ODOR;  % 4 s
                trial_type{end+1,1}  = 'fixation';
                run_col(end+1,1)     = r;
                trial_col(end+1,1)   = tr;
                cue_idx_col(end+1,1) = cuei;
                odor_num_col{end+1,1}= 'n/a';
                response_label{end+1,1} = 'n/a';
                rt_col(end+1,1)      = NaN;
                acc_col(end+1,1)     = NaN;
            end

            % ---- advance time to next trial ----
            t = t + ACTIVE_9 + iti;
        end

        % Assemble table (types chosen to accommodate 'n/a')
        events = table( ...
            onsets, durations, trial_type, run_col, trial_col, ...
            cue_idx_col, odor_num_col, response_label, rt_col, acc_col, ...
            'VariableNames', {'onset','duration','trial_type','run','trial', ...
                              'cue_idx','odor_num','response_label','rt','acc'} );

        % Save
        out_tsv = fullfile(outdir, sprintf('%s_task-odorpred_run-%d_events.tsv', SubID, r));
        writetable(events, out_tsv, 'FileType','text','Delimiter','\t');
        fprintf('✅ Saved %s (rows=%d)\n', out_tsv, height(events));

        % Sanity print
        n_pred_rows = sum(strcmp(events.trial_type,'prediction'));
        n_btn_rows  = sum(strcmp(events.trial_type,'button'));
        n_miss      = sum(strcmp(events.trial_type,'prediction') & strcmp(events.response_label,'n/a'));
        run_secs    = t; % includes lead-in + all trials + ITIs
        fprintf('   predictions=%d, buttons=%d, misses=%d, est_len=%.1fs (~%.1f vols @ TR=%.1fs)\n', ...
                n_pred_rows, n_btn_rows, n_miss, run_secs, run_secs/TR, TR);
    end
end
