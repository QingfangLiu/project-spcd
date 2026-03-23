
%% ========================================================================
%  Pre-meal LSS: single-event GLMs for cue images (2 mm smooth)
%
%  This script runs LSS (least-squares single) mini-GLMs to produce single-
%  image (single-trial) beta estimates for cue images in the premeal probe
%  task. Each mini-GLM fits one "target" image vs. all other images grouped.
%
%  Mini-GLM design (per target image):
%    1) target image (3.0 s)
%    2) all other images (grouped; 3.0 s each)
%    3) decision period (2.0 s; all trials)
%    4) button sticks (0 s) — split by response hand: button_left / button_right
%
%  IMPORTANT BEHAVIORAL / MODELING CHANGES (compared to prior version):
%    - Button presses are split by the 'response_hand' column in the events.tsv
%      and modeled as two separate stick regressors ('button_left', 'button_right')
%      when that information exists. This helps remove lateralized motor
%      variance from cue betas.
%    - The script **keeps** SPM outputs for each mini-GLM in a compact folder:
%         <GLM_root>/<subID>/miniGLMs/run## _ img###/
%      Each mini-GLM folder contains at least the copied SPM.mat and a single
%      residual file named ResMS.nii (or the first Res*.nii found renamed to
%      ResMS.nii). This keeps one ResMS.nii per mini-GLM and avoids filename
%      collisions while saving space.
%
%  Inputs:
%    - NIfTI:  SPCD_data/nii/<subID>/func_PAID/premeal/s2w2Scan2_premeal_run_<r>_*_PAID.nii
%    - Events: SPCD_data/nii/<subID>/func_PAID/premeal/<subID>_task-premeal_run-<r>_events.tsv
%    - Nuisance: SPCD_data/NRegressor/<subID>/nuisance_regressors_premeal_run_<r>_zscored.txt
%
%  Outputs (per subject):
%    - Single-trial beta NIfTIs (moved to):
%        SPCD_data/GLM/premeal_trialwise_2mm/<subID>/beta_run<r>_img<k>.nii
%    - Mini-GLM output folders (compact, one folder per mini-GLM):
%        SPCD_data/GLM/premeal_trialwise_2mm/<subID>/miniGLMs/run<rr>_img<kk>/SPM.mat
%        SPCD_data/GLM/premeal_trialwise_2mm/<subID>/miniGLMs/run<rr>_img<kk>/ResMS.nii
%      (these are created only when those files exist in the outdir; see note
%       above if you want full residual volumes instead)
%    - Beta index CSV for each run:
%        SPCD_data/GLM/premeal_trialwise_2mm/<subID>/<subID>_premeal_run-<r>_beta_index.csv
%      The CSV maps each beta file to run / trial / img_index / onset / trial_type / ...
%
%  Notes:
%    - Skips sub-36 run 3 (bad MRI data)
%    - Uses SPM12
%
%  Author / Date: (edit as appropriate)
%    Qingfang Liu / adapted 2026-03-10 (notes added to reflect button-splitting
%    and compact residual saving behavior)
%% ========================================================================

clear; clc;
addpath('/Users/liuq13/Applications/spm12/')

% ---- PATHS ----
HomeDir = '/Users/liuq13/SPCD_data';
niiRoot = fullfile(HomeDir, 'nii');
glmRoot = fullfile(HomeDir, 'GLM', 'premeal_trialwise_2mm');
taskname = 'premeal';

if ~exist(glmRoot, 'dir'), mkdir(glmRoot); end

TR = 1.5;

% ---- SUBJECTS ----
subs = dir(fullfile(niiRoot, 'sub-*')); subs = subs([subs.isdir]);

for s = 1:numel(subs)
    SubID = subs(s).name;
    fprintf('\n=== LSS | %s ===\n', SubID);

    % where betas will live (flat folder per subject)
    subj_beta_dir = fullfile(glmRoot, SubID); 
    if ~exist(subj_beta_dir, 'dir'), mkdir(subj_beta_dir); end

    % Simple run list: 1..3, but only keep those that actually have events files
    run_list = 1:3;
    ev_dir = fullfile(niiRoot, SubID, 'func_PAID', taskname);
    has_ev = arrayfun(@(r) exist(fullfile(ev_dir, ...
        sprintf('%s_task-%s_run-%d_events.tsv', SubID, taskname, r)), 'file')==2, run_list);
    run_list = run_list(has_ev);

    for r = run_list(:)'
        % hard skip: sub-36 run 3
        if strcmp(SubID,'sub-36') && r==3
            fprintf('  ⚠  Skipping %s run %d (known bad MRI)\n', SubID, r);
            continue
        end

        % index table to map betas back to events
        beta_log = table('Size',[0 10], ...
      'VariableTypes', {'string','double','double','double','double','double','string','double','double','double'}, ...
      'VariableNames', {'beta_file','run','event_index','trial','img_index','onset','trial_type', ...
                        'cue_index','odor_index','set'});

        % ---- Load events ----
        tsv_file = fullfile(ev_dir, sprintf('%s_task-%s_run-%d_events.tsv', SubID, taskname, r));
        if ~exist(tsv_file,'file'); fprintf('  ❌ Missing events: %s\n', tsv_file); continue; end
        E = readtable(tsv_file, 'FileType','text','Delimiter','\t');

        % Identify event sets
        is_img   = ismember(E.trial_type, {'cue_img1','cue_img2'});
        is_dec   = strcmp(E.trial_type, 'decision');
        is_btn   = strcmp(E.trial_type, 'button');

        img_on   = E.onset(is_img);
        img_dur  = E.duration(is_img);   % should be 3s each
        img_type = E.trial_type(is_img); % cue_img1 / cue_img2
        img_trial= E.trial(is_img);
        img_idx  = E.img_index(is_img);  % 1 or 2

        cue_index_img = E.cue_index(is_img);
        odor_index_img = E.odor_index(is_img);
        set_img = E.set(is_img);

        dec_on   = E.onset(is_dec);
        dec_dur  = E.duration(is_dec);   % 2s

        % ---- split button onsets by response hand (left / right) ----
        
        btn_on_all   = E.onset(is_btn);   % all button onsets

        hand_col = 'response_hand';
        raw_hand = E.(hand_col)(is_btn);
        hand_str = string(raw_hand);     % convert to string array; 'NaN' or '' possible

        % normalize common encodings (case-insensitive)
        is_left  = strcmpi(hand_str, 'left');
        is_right = strcmpi(hand_str, 'right');

        % Some rows may contain 'n/a' or missing; keep those out of left/right
        is_left(hand_str=="n/a" | hand_str=="NaN" | hand_str=="") = false;
        is_right(hand_str=="n/a" | hand_str=="NaN" | hand_str=="") = false;

        btn_on_left  = btn_on_all(is_left);
        btn_on_right = btn_on_all(is_right);

        n_img = numel(img_on);
        if n_img == 0
            fprintf('  ❌ No image events in %s run %d\n', SubID, r);
            continue
        end

        % ---- Locate s2w2 4D NIfTI ----
        scandir = fullfile(niiRoot, SubID, 'func_PAID', taskname);
        nii_struct = dir(fullfile(scandir, sprintf('s2w2Scan2_%s_run_%d_*_PAID.nii', taskname, r)));
        if isempty(nii_struct)
            fprintf('  ❌ No s2w2 .nii for run %d in %s\n', r, scandir);
            continue
        end
        nii_file = fullfile(scandir, nii_struct(1).name);
        vols = cellstr(spm_select('expand', nii_file));

        % ---- Nuisance regressors ----
        regfile = fullfile(HomeDir, 'NRegressor', SubID, ...
            sprintf('nuisance_regressors_%s_run_%d_zscored.txt', taskname, r));
        if ~exist(regfile, 'file')
            fprintf('  ❌ Missing nuisance: %s\n', regfile); 
            continue
        end

        % ---- Loop over image events (LSS mini-GLMs) ----
        for k = 1:n_img
            target_on   = img_on(k);
            target_dur  = img_dur(k); % 3
            other_mask  = true(n_img,1); other_mask(k) = false;

            other_on    = img_on(other_mask);
            other_dur   = img_dur(other_mask); % all 3s

            outdir = fullfile(glmRoot, 'work', SubID, sprintf('run%02d_img%03d', r, k));
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
            matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.1;  
            matlabbatch{1}.spm.stats.fmri_spec.global = 'None';

            % Target regressor
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name     = 'target_img';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset    = target_on;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = target_dur;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod     = struct('name',{},'param',{},'poly',{});

            % Other-images (grouped)
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name     = 'other_images';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset    = other_on';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = other_dur';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod     = struct('name',{},'param',{},'poly',{});

            % Decision (all trials)
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).name     = 'decision';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).onset    = dec_on';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).duration = dec_dur';
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).tmod     = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).pmod     = struct('name',{},'param',{},'poly',{});

            % === Append button condition(s) SAFELY using dynamic index ===
            next_cond = numel(matlabbatch{1}.spm.stats.fmri_spec.sess.cond) + 1;

            % Button sticks of left
            if ~isempty(btn_on_left)
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).name     = 'button_left';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).onset    = btn_on_left';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).duration = zeros(size(btn_on_left'))';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).tmod     = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).pmod     = struct('name',{},'param',{},'poly',{});
                next_cond = next_cond + 1;
            end

            % Button sticks of right
            if ~isempty(btn_on_right)
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).name     = 'button_right';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).onset    = btn_on_right';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).duration = zeros(size(btn_on_right'))';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).tmod     = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).pmod     = struct('name',{},'param',{},'poly',{});
                next_cond = next_cond + 1;
            end

            % Nuisance + HPF
            matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {regfile};
            matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;

            % Estimation
            matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(outdir, 'SPM.mat')};

            % Run
            spm_jobman('run', matlabbatch);

            % Move beta_0001 to subject folder with clean name
            src_beta = fullfile(outdir, 'beta_0001.nii');
            beta_name = sprintf('beta_run%d_img%03d.nii', r, k);
            dst_beta = fullfile(subj_beta_dir, beta_name);
            movefile(src_beta, dst_beta);

            % Log mapping
            beta_log = [beta_log; { ...
                beta_name, ...          % beta_file
                r, ...                  % run
                k, ...                  % event_index within image-events
                img_trial(k), ...       % trial
                img_idx(k), ...         % img_index (1 or 2)
                target_on, ...          % onset (for this image event)
                img_type(k), ...        % 'cue_img1' / 'cue_img2'
                cue_index_img(k), ...   % new
                odor_index_img(k), ...  % new
                set_img(k) ...          % new
            }];

            % Move the entire mini-GLM folder to subject-level storage for later access
            sess_store_dir = fullfile(subj_beta_dir, 'miniGLMs'); % e.g., SPCD_data/GLM/.../sub-XX/miniGLMs/
            if ~exist(sess_store_dir, 'dir'), mkdir(sess_store_dir); end
            
            % Create a unique folder for this mini-GLM (keeps ResMS.nii name identical inside)
            mini_name = sprintf('run%02d_img%03d', r, k);
            mini_dir  = fullfile(sess_store_dir, mini_name);
            if ~exist(mini_dir, 'dir'), mkdir(mini_dir); end

            % Copy SPM.mat (if present)
            src_spm = fullfile(outdir, 'SPM.mat');
            if exist(src_spm, 'file')
                copyfile(src_spm, fullfile(mini_dir, 'SPM.mat'));
            end

            % Copy ResMS.nii (if present). This keeps the filename exactly 'ResMS.nii' inside mini_dir.
            src_resms = fullfile(outdir, 'ResMS.nii');
            if exist(src_resms, 'file')
                copyfile(src_resms, fullfile(mini_dir, 'ResMS.nii'));
            end

            % Remove the original mini-GLM folder to save space (recursively). If copy failed, keep original.
            try
            % Only remove if SPM.mat or ResMS.nii was successfully copied (safety)
                if exist(fullfile(mini_dir, 'SPM.mat'),'file') || exist(fullfile(mini_dir, 'ResMS.nii'),'file')
                    rmdir(outdir, 's');
                else
                    warning('Did not remove %s because nothing was copied into %s', outdir, mini_dir);
                end
            catch ME
                warning('Could not remove %s: %s', outdir, ME.message);
            end

        end % end of mini-GLMs

        % Save index CSV for this subject/run
        out_csv = fullfile(subj_beta_dir, sprintf('%s_premeal_run-%d_beta_index.csv', SubID, r));
        writetable(beta_log, out_csv);
        fprintf('✅ Saved beta index for run %d: %s\n', r, out_csv);

    end % end of run

end
