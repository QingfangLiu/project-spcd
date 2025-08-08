
%% ========================================================================
%  Post-meal Probe Event File Generator
%
%  This script generates BIDS-like events.tsv files for the Post-meal probe
%  task for each subject in SPCD_data/nii/sub-*.
%
%  Input:
%    - Behavioral CSV: project-SPCD/beh_processed_subs/<subID>/post_meal_dat.csv
%      Required columns: Run, trial, ISIDur, ITIDur, resp, rt
%    - Functional data folder: SPCD_data/nii/<subID>/func_PAID/postmeal/
%
%  Output:
%    - One events.tsv per available run:
%        sub-XX_task-postmeal_run-<r>_events.tsv
%      Saved in: SPCD_data/nii/<subID>/func_PAID/postmeal/
%
%  Event structure per trial (absolute onsets in seconds):
%    1) cue_img1   (3s) at trial start
%    2) cue_img2   (3s) after ISIDur
%    3) decision   (2s) after cue_img2
%    4) button     (0s stick) at decision_onset + rt, only if response present
%
%  Timing:
%    - Run starts at t = 10s (pre-run blank)
%    - cue1 → ISI(ISIDur) → cue2 → decision(2s) → ITI(ITIDur)
%    - resp: 0 = left, 2 = right, NaN = no response
%
%  Additional:
%    - Detects available runs from CSV
%    - Skips button events if no response
%    - Flags RTs outside (0, 2] seconds
%    - Prints basic run statistics (buttons, misses, est. run length)
%
%% ========================================================================


%%

clear; clc;

% ---------- USER CONFIG ----------
HomeDir = '/Users/liuq13/SPCD_data';
niiDir = fullfile(HomeDir, 'nii');
behDir  = '/Users/liuq13/project-SPCD/beh_processed_subs';

subs = dir(fullfile(niiDir,'sub-*')); subs = subs([subs.isdir]);

% Cue timing
BLANK   = 10;   % pre-run blank (s)
DUR_I1  = 3;    % cue1 duration (s)
DUR_I2  = 3;    % cue2 duration (s)
DUR_DEC = 2;    % decision duration (s)
TR      = 1.5;  % for sanity reporting only

for s = 1:numel(subs)
    SubID = subs(s).name;
    in_csv = fullfile(behDir, SubID, 'post_meal_dat.csv');
    out_dir = fullfile(niiDir, SubID, 'func_PAID', 'postmeal');
    if ~exist(out_dir,'dir'), mkdir(out_dir); end

    if ~exist(in_csv,'file')
        fprintf('❌ Missing CSV for %s: %s\n', SubID, in_csv); 
        continue
    end

    T = readtable(in_csv, 'VariableNamingRule','preserve');

    % Required columns
    reqCols = {'Run','trial','ISIDur','ITIDur','resp','rt'};
    miss = setdiff(reqCols, T.Properties.VariableNames);
    if ~isempty(miss)
        fprintf('❌ %s missing columns: %s\n', SubID, strjoin(miss, ', ')); 
        continue
    end

    run_list = unique(T.Run(~isnan(T.Run)))';
    fprintf('\n=== %s | postmeal events | runs: %s ===\n', SubID, sprintf('%g ', run_list));

    for r = run_list
        R = sortrows(T(T.Run==r,:), 'trial');
        t = BLANK;

        onsets = []; durations = []; run_col = []; trial_col = []; img_idx = [];
        trial_type = {}; response_time = []; response_hand = {};
        bad_rt_trials = [];
        cue_index = [];
        odor_index = [];
        set_col   = [];

        for i = 1:height(R)
            % ---- cue1 ----
            cue1_on = t;
            onsets(end+1,1)   = cue1_on; 
            durations(end+1,1)= DUR_I1;
            trial_type{end+1,1} = 'cue_img1';
            run_col(end+1,1)  = r;
            trial_col(end+1,1)= R.trial(i);
            img_idx(end+1,1)  = 1;
            response_time(end+1,1) = NaN;
            response_hand{end+1,1} = 'n/a';
            cue_index(end+1,1) = R.cue_left_idx(i);
            odor_index(end+1,1) = R.odor_left(i);
            set_col(end+1,1)    = R.set(i);

            % ---- cue2 ----
            cue2_on = cue1_on + DUR_I1 + R.ISIDur(i);
            onsets(end+1,1)   = cue2_on;
            durations(end+1,1)= DUR_I2;
            trial_type{end+1,1} = 'cue_img2';
            run_col(end+1,1)  = r;
            trial_col(end+1,1)= R.trial(i);
            img_idx(end+1,1)  = 2;
            response_time(end+1,1) = NaN;
            response_hand{end+1,1} = 'n/a';
            cue_index(end+1,1) = R.cue_right_idx(i);
            odor_index(end+1,1) = R.odor_right(i);
            set_col(end+1,1)    = R.set(i);

            % ---- decision (always) ----
            dec_on = cue2_on + DUR_I2;
            onsets(end+1,1)   = dec_on;
            durations(end+1,1)= DUR_DEC;
            trial_type{end+1,1} = 'decision';
            run_col(end+1,1)  = r;
            trial_col(end+1,1)= R.trial(i);
            img_idx(end+1,1)  = NaN;
            response_time(end+1,1) = NaN;
            response_hand{end+1,1} = 'n/a';
            cue_index(end+1,1) = NaN;
            odor_index(end+1,1) = NaN;
            set_col(end+1,1)    = NaN;

            % ---- button (only if response present) ----
            if ~isnan(R.resp(i))
                rt_i = R.rt(i);
                if ~(rt_i > 0 && rt_i <= DUR_DEC), bad_rt_trials(end+1) = R.trial(i); end %#ok<SAGROW>
                btn_on = dec_on + rt_i;

                onsets(end+1,1)   = btn_on;
                durations(end+1,1)= 0;
                trial_type{end+1,1} = 'button';
                run_col(end+1,1)  = r;
                trial_col(end+1,1)= R.trial(i);
                img_idx(end+1,1)  = NaN;
                response_time(end+1,1) = rt_i;
                if R.resp(i)==0
                    response_hand{end+1,1} = 'left';
                elseif R.resp(i)==2
                    response_hand{end+1,1} = 'right';
                else
                    response_hand{end+1,1} = 'n/a';
                end
                cue_index(end+1,1) = NaN;
                odor_index(end+1,1) = NaN;
                set_col(end+1,1)    = NaN;
            end

            % advance time
            t = dec_on + DUR_DEC + R.ITIDur(i);
        end

        % Assemble and save
        events = table(onsets, durations, trial_type, run_col, trial_col, img_idx, response_time, response_hand, cue_index, odor_index, set_col, ...
                       'VariableNames', {'onset','duration','trial_type','run','trial','img_index','response_time','response_hand','cue_index','odor_index','set'});
        out_tsv = fullfile(out_dir, sprintf('%s_task-postmeal_run-%d_events.tsv', SubID, r));
        writetable(events, out_tsv, 'FileType','text','Delimiter','\t');
        fprintf('✅ Saved %s (rows=%d)\n', out_tsv, height(events));

        % Sanity print
        n_dec  = sum(strcmp(events.trial_type,'decision'));
        n_btn  = sum(strcmp(events.trial_type,'button'));
        n_miss = n_dec - n_btn;
        run_secs = t;
        fprintf('   decisions=%d, buttons=%d, misses=%d, est_len=%.1fs (~%.1f vols @ TR=%.1fs)\n', ...
                n_dec, n_btn, n_miss, run_secs, run_secs/TR, TR);
        if ~isempty(bad_rt_trials)
            fprintf('   ⚠ RT outside (0, %.1f] on trials: %s\n', DUR_DEC, mat2str(bad_rt_trials));
        end
    end
end