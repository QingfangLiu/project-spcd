% run_probe_imagecategory_univariate_group.m
%
% Whole-brain univariate control analysis for probe image category:
%   character cues vs. place cues
%
% Inputs are cue-wise beta maps from:
%   SPCD_data/GLM/premeal_cuewise_2mm/<sub>/both/
%   SPCD_data/GLM/postmeal_cuewise_2mm/<sub>/both/
%
% The subject-specific character/place cue assignment is read from the
% repo-level subject_info.csv pairing column:
%   pairing = c: conditioned/even cues are characters
%   pairing = p: conditioned/even cues are places
%
% This script:
%   1) Builds subject-level beta contrast maps for each session.
%   2) Builds collapsed subject maps by averaging premeal and postmeal maps.
%   3) Runs SPM one-sample group tests for both contrast directions.
%   4) Uses the explicit gray-matter mask gm_0.1.nii.
%   5) Skips outputs that already exist.

clear; clc;

%% ---------- Settings ----------
addpath('/Users/liuq13/Applications/spm12/');

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(script_dir));
subject_info_file = fullfile(repo_root, 'subject_info.csv');

base_dir = '/Users/liuq13/SPCD_data';
glm_root = fullfile(base_dir, 'GLM');

out_root = fullfile(glm_root, 'probe_cuewise_univariate_imagecategory');
subject_out_root = fullfile(out_root, 'subject_contrasts');
group_out_root = fullfile(out_root, 'group_tmaps');
tmp_root = fullfile(out_root, '_tmp_spm_work');

mask_file = fullfile(base_dir, 'masks', 'gm_0.1.nii');

sessions = {'premeal', 'postmeal'};
session_folders = {'premeal_cuewise_2mm', 'postmeal_cuewise_2mm'};
mode = 'both';

skip_subs = {'sub-19', 'sub-33'};
runs = 1:3;

even_cues = [2 4 6 8 10 12];     % conditioned cues
odd_cues = [1 3 5 7 9 11];       % preconditioned cues

contrast_tags = {'character_gt_place', 'place_gt_character'};

fwhm = 6; % smoothing kernel in mm for group tests

ensure_dir(out_root);
ensure_dir(subject_out_root);
ensure_dir(group_out_root);
ensure_dir(tmp_root);

%% ---------- Init SPM ----------
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

if ~exist(mask_file, 'file')
    error('Mask file not found: %s', mask_file);
end

if ~exist(subject_info_file, 'file')
    error('Subject info file not found: %s', subject_info_file);
end

subject_info = read_subject_info(subject_info_file);

Vm = spm_vol(mask_file);
M = spm_read_vols(Vm) > 0;

fprintf('\n============================================================\n');
fprintf('Probe image-category univariate analysis\n');
fprintf('Input GLM root:     %s\n', glm_root);
fprintf('Output root:        %s\n', out_root);
fprintf('Mode folder:        %s\n', mode);
fprintf('Subject info file:  %s\n', subject_info_file);
fprintf('Mask:               %s\n', mask_file);
fprintf('============================================================\n');

%% ---------- Part 1: Subject-level session contrasts ----------
for si = 1:numel(sessions)
    session_tag = sessions{si};
    session_folder = session_folders{si};
    glm_session_dir = fullfile(glm_root, session_folder);
    subject_session_out = fullfile(subject_out_root, session_tag);
    ensure_dir(subject_session_out);

    fprintf('\n------------------------------------------------------------\n');
    fprintf('Subject contrasts | session=%s\n', session_tag);
    fprintf('Input folder: %s\n', glm_session_dir);
    fprintf('------------------------------------------------------------\n');

    if ~exist(glm_session_dir, 'dir')
        fprintf('Input folder not found, skipping session: %s\n', glm_session_dir);
        continue;
    end

    subjects = list_subjects(glm_session_dir, skip_subs);

    for s = 1:numel(subjects)
        sub = subjects{s};
        sub_mode_dir = fullfile(glm_session_dir, sub, mode);

        if ~exist(sub_mode_dir, 'dir')
            fprintf('[SKIP] %s %s: mode folder not found: %s\n', ...
                sub, session_tag, sub_mode_dir);
            continue;
        end

        try
            [character_cues, place_cues, pairing] = ...
                image_category_cues_for_subject(sub, subject_info, even_cues, odd_cues);
        catch ME
            fprintf('[SKIP] %s %s: %s\n', sub, session_tag, ME.message);
            continue;
        end

        out_char = subject_contrast_path( ...
            subject_session_out, sub, session_tag, ...
            'character_gt_place', mode);
        out_place = subject_contrast_path( ...
            subject_session_out, sub, session_tag, ...
            'place_gt_character', mode);

        if exist(out_char, 'file') && exist(out_place, 'file')
            fprintf('[SKIP] %s %s: subject contrast maps already exist.\n', ...
                sub, session_tag);
            continue;
        end

        try
            [char_minus_place, Vref, n_runs_used, n_char, n_place] = ...
                build_subject_contrast( ...
                    sub_mode_dir, session_tag, runs, ...
                    character_cues, place_cues, M);
        catch ME
            fprintf('[SKIP] %s %s: %s\n', sub, session_tag, ME.message);
            continue;
        end

        if isempty(char_minus_place)
            fprintf('[SKIP] %s %s: no complete runs found.\n', sub, session_tag);
            continue;
        end

        if ~isequal(size(char_minus_place), size(M))
            error('Contrast map and GM mask dimensions do not match for %s %s.', ...
                sub, session_tag);
        end

        char_minus_place(~M) = NaN;

        fprintf(['[WRITE] %s %s: pairing=%s, %d complete runs, ' ...
            '%d character maps, %d place maps.\n'], ...
            sub, session_tag, pairing, n_runs_used, n_char, n_place);

        if ~exist(out_char, 'file')
            write_nifti_float(char_minus_place, Vref, out_char, ...
                sprintf('%s %s character > place', sub, session_tag));
        else
            fprintf('  Existing, not overwritten: %s\n', out_char);
        end

        if ~exist(out_place, 'file')
            write_nifti_float(-char_minus_place, Vref, out_place, ...
                sprintf('%s %s place > character', sub, session_tag));
        else
            fprintf('  Existing, not overwritten: %s\n', out_place);
        end
    end
end

%% ---------- Part 2: Collapsed subject contrasts ----------
fprintf('\n------------------------------------------------------------\n');
fprintf('Collapsed subject contrasts | average premeal and postmeal\n');
fprintf('------------------------------------------------------------\n');

collapsed_out = fullfile(subject_out_root, 'collapsed');
ensure_dir(collapsed_out);

make_collapsed_subject_maps( ...
    subject_out_root, collapsed_out, mode, contrast_tags, skip_subs, M);

%% ---------- Part 3: Group tests ----------
fprintf('\n------------------------------------------------------------\n');
fprintf('Group one-sample tests\n');
fprintf('------------------------------------------------------------\n');

group_sessions = {'premeal', 'postmeal', 'collapsed'};

for si = 1:numel(group_sessions)
    session_tag = group_sessions{si};
    subject_session_dir = fullfile(subject_out_root, session_tag);

    if ~exist(subject_session_dir, 'dir')
        fprintf('[SKIP] Group session folder not found: %s\n', subject_session_dir);
        continue;
    end

    for di = 1:numel(contrast_tags)
        contrast_tag = contrast_tags{di};

        run_group_one_sample( ...
            subject_session_dir, group_out_root, session_tag, contrast_tag, ...
            mode, skip_subs, M, mask_file, fwhm, tmp_root);
    end
end

fprintf('\nAll done.\n');

%% ========================================================================
% Local helper functions
% ========================================================================

function ensure_dir(p)
    if ~exist(p, 'dir')
        mkdir(p);
    end
end

function info = read_subject_info(subject_info_file)
    T = readtable(subject_info_file);
    required_cols = {'subject', 'pairing'};
    for i = 1:numel(required_cols)
        if ~ismember(required_cols{i}, T.Properties.VariableNames)
            error('subject_info.csv is missing required column: %s', required_cols{i});
        end
    end

    subjects = table_column_to_cellstr(T.subject);
    pairings = table_column_to_cellstr(T.pairing);
    for i = 1:numel(pairings)
        pairings{i} = lower(strtrim(pairings{i}));
    end

    bad = ~ismember(pairings, {'c', 'p'});
    if any(bad)
        error('subject_info.csv has invalid pairing value(s): %s', ...
            strjoin(unique(pairings(bad)), ', '));
    end

    info = containers.Map(subjects, pairings);
end

function values = table_column_to_cellstr(col)
    if iscell(col)
        values = cellfun(@char, col, 'UniformOutput', false);
    elseif isstring(col)
        values = cellstr(col);
    elseif isnumeric(col)
        values = cellstr(num2str(col));
    else
        values = cellstr(string(col));
    end
end

function [character_cues, place_cues, pairing] = ...
    image_category_cues_for_subject(sub, subject_info, even_cues, odd_cues)

    if ~isKey(subject_info, sub)
        error('subject not found in subject_info.csv');
    end

    pairing = subject_info(sub);

    switch pairing
        case 'c'
            character_cues = even_cues;
            place_cues = odd_cues;
        case 'p'
            character_cues = odd_cues;
            place_cues = even_cues;
        otherwise
            error('invalid pairing for %s: %s', sub, pairing);
    end
end

function subjects = list_subjects(glm_session_dir, skip_subs)
    d = dir(fullfile(glm_session_dir, 'sub-*'));
    d = d([d.isdir]);
    subjects = {d.name};
    keep = true(size(subjects));
    for i = 1:numel(subjects)
        keep(i) = ~any(strcmp(subjects{i}, skip_subs));
    end
    subjects = sort(subjects(keep));
end

function out_path = subject_contrast_path(out_dir, sub, session_tag, contrast_tag, mode)
    out_path = fullfile(out_dir, sprintf( ...
        '%s_session-%s_contrast-%s_mode-%s.nii', ...
        sub, session_tag, contrast_tag, mode));
end

function [char_minus_place, Vref, n_runs_used, n_char, n_place] = ...
    build_subject_contrast(sub_mode_dir, session_tag, runs, character_cues, place_cues, M)

    char_sum = [];
    place_sum = [];
    Vref = [];
    n_char = 0;
    n_place = 0;
    n_runs_used = 0;

    for r = runs
        char_files = cue_files_for_run(sub_mode_dir, session_tag, r, character_cues);
        place_files = cue_files_for_run(sub_mode_dir, session_tag, r, place_cues);

        if any(cellfun(@isempty, char_files)) || any(cellfun(@isempty, place_files))
            fprintf('  Missing one or more cue beta maps in run %02d, skipping run.\n', r);
            continue;
        end

        n_runs_used = n_runs_used + 1;
        [char_sum, Vref, n_char] = add_maps(char_files, char_sum, Vref, n_char, M);
        [place_sum, Vref, n_place] = add_maps(place_files, place_sum, Vref, n_place, M);
    end

    if n_char == 0 || n_place == 0
        char_minus_place = [];
        return;
    end

    char_minus_place = (char_sum ./ n_char) - (place_sum ./ n_place);
end

function files = cue_files_for_run(sub_mode_dir, session_tag, run_num, cue_indices)
    files = cell(numel(cue_indices), 1);
    for i = 1:numel(cue_indices)
        f = fullfile(sub_mode_dir, sprintf( ...
            'beta_task-%s_run-%02d_cue-%02d.nii', ...
            session_tag, run_num, cue_indices(i)));
        if exist(f, 'file')
            files{i} = f;
        elseif exist([f '.gz'], 'file')
            files{i} = [f '.gz'];
        else
            files{i} = '';
        end
    end
end

function [sum_data, Vref, n_maps] = add_maps(files, sum_data, Vref, n_maps, M)
    for i = 1:numel(files)
        [Y, V] = read_nifti_maybe_gz(files{i});

        if isempty(Vref)
            Vref = V;
        end

        if ~isequal(size(Y), size(M))
            error('Input map dimensions do not match GM mask: %s', files{i});
        end

        if isempty(sum_data)
            sum_data = zeros(size(Y));
        elseif ~isequal(size(Y), size(sum_data))
            error('Input map dimensions differ within subject: %s', files{i});
        end

        sum_data = sum_data + Y;
        n_maps = n_maps + 1;
    end
end

function [Y, V] = read_nifti_maybe_gz(path_in)
    temp_dir = '';
    path_to_read = path_in;

    if numel(path_in) > 3 && strcmp(path_in(end-2:end), '.gz')
        temp_dir = tempname;
        mkdir(temp_dir);
        gunzip(path_in, temp_dir);
        [~, base_gz, ~] = fileparts(path_in);
        path_to_read = fullfile(temp_dir, base_gz);
    end

    V = spm_vol(path_to_read);
    Y = spm_read_vols(V);

    if ~isempty(temp_dir) && exist(temp_dir, 'dir')
        try
            rmdir(temp_dir, 's');
        catch
        end
    end
end

function write_nifti_float(data, Vref, out_path, description)
    ensure_dir(fileparts(out_path));
    Vout = Vref;
    Vout.fname = out_path;
    Vout.dt = [16 0]; % float32
    Vout.pinfo = [1; 0; 0];
    Vout.descrip = description;
    spm_write_vol(Vout, data);
    fprintf('  Saved: %s\n', out_path);
end

function make_collapsed_subject_maps(subject_out_root, collapsed_out, mode, contrast_tags, skip_subs, M)
    pre_dir = fullfile(subject_out_root, 'premeal');
    post_dir = fullfile(subject_out_root, 'postmeal');

    if ~exist(pre_dir, 'dir') || ~exist(post_dir, 'dir')
        fprintf('[SKIP] Missing premeal or postmeal subject contrast folder.\n');
        return;
    end

    base_tag = 'character_gt_place';
    pre_files = dir(fullfile(pre_dir, sprintf( ...
        'sub-*_session-premeal_contrast-%s_mode-%s.nii', base_tag, mode)));

    for i = 1:numel(pre_files)
        sub = parse_subject_id(pre_files(i).name);
        if isempty(sub) || any(strcmp(sub, skip_subs))
            continue;
        end

        pre_path = fullfile(pre_files(i).folder, pre_files(i).name);
        post_path = subject_contrast_path(post_dir, sub, 'postmeal', base_tag, mode);
        if ~exist(post_path, 'file')
            fprintf('[SKIP] %s collapsed: missing postmeal contrast.\n', sub);
            continue;
        end

        out_char = subject_contrast_path(collapsed_out, sub, 'collapsed', ...
            'character_gt_place', mode);
        out_place = subject_contrast_path(collapsed_out, sub, 'collapsed', ...
            'place_gt_character', mode);

        if exist(out_char, 'file') && exist(out_place, 'file')
            fprintf('[SKIP] %s collapsed: maps already exist.\n', sub);
            continue;
        end

        [Ypre, Vref] = read_nifti_maybe_gz(pre_path);
        [Ypost, ~] = read_nifti_maybe_gz(post_path);

        if ~isequal(size(Ypre), size(Ypost)) || ~isequal(size(Ypre), size(M))
            error('Collapsed map dimensions do not match for %s.', sub);
        end

        Yavg = (Ypre + Ypost) ./ 2;
        Yavg(~M) = NaN;

        if ~exist(out_char, 'file')
            write_nifti_float(Yavg, Vref, out_char, ...
                sprintf('%s collapsed character > place', sub));
        else
            fprintf('  Existing, not overwritten: %s\n', out_char);
        end

        if ~exist(out_place, 'file')
            write_nifti_float(-Yavg, Vref, out_place, ...
                sprintf('%s collapsed place > character', sub));
        else
            fprintf('  Existing, not overwritten: %s\n', out_place);
        end
    end

    fprintf('Collapsed contrast directions defined: %s, %s\n', ...
        contrast_tags{1}, contrast_tags{2});
end

function sub = parse_subject_id(fname)
    tok = regexp(fname, '^(sub-\d+)_', 'tokens', 'once');
    if isempty(tok)
        sub = '';
    else
        sub = tok{1};
    end
end

function run_group_one_sample(subject_session_dir, group_out_root, session_tag, contrast_tag, mode, skip_subs, M, mask_file, fwhm, tmp_root)
    pat = sprintf('sub-*_session-%s_contrast-%s_mode-%s.nii', ...
        session_tag, contrast_tag, mode);
    files = dir(fullfile(subject_session_dir, pat));

    keep = true(size(files));
    for i = 1:numel(files)
        sub = parse_subject_id(files(i).name);
        keep(i) = ~isempty(sub) && ~any(strcmp(sub, skip_subs));
    end
    files = files(keep);

    fprintf('\nGroup test | session=%s | contrast=%s\n', session_tag, contrast_tag);
    fprintf('Pattern: %s\n', pat);
    fprintf('Found %d subject maps.\n', numel(files));

    if isempty(files)
        fprintf('[SKIP] No subject maps for group test.\n');
        return;
    end

    out_dir = fullfile(group_out_root, session_tag);
    ensure_dir(out_dir);

    out_name = sprintf('group_session-%s_contrast-%s.nii.gz', ...
        session_tag, contrast_tag);
    out_path_gz = fullfile(out_dir, out_name);
    out_path_nii = fullfile(out_dir, erase(out_name, '.gz'));

    if exist(out_path_gz, 'file') || exist(out_path_nii, 'file')
        fprintf('[SKIP] Group output exists: %s\n', out_path_gz);
        return;
    end

    tmp_dir = tempname(tmp_root);
    ensure_dir(tmp_dir);
    spm_input_dir = fullfile(tmp_dir, 'spm_inputs');
    ensure_dir(spm_input_dir);

    smoothed_files = cell(0, 1);

    for i = 1:numel(files)
        src_path = fullfile(files(i).folder, files(i).name);
        [Y, V] = read_nifti_maybe_gz(src_path);

        if ~isequal(size(Y), size(M))
            error('Subject map dimensions do not match GM mask: %s', src_path);
        end

        Y(~M) = NaN;

        base_name = regexprep(files(i).name, '\.nii(\.gz)?$', '');
        masked_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
        smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);

        Vc = V;
        Vc.fname = masked_path;
        Vc.dt = [16 0];
        Vc.pinfo = [1; 0; 0];
        spm_write_vol(Vc, Y);

        fprintf('  Mask->NaN + smooth: %s\n', base_name);
        spm_smooth(masked_path, smooth_path, [fwhm fwhm fwhm]);
        smoothed_files{end+1, 1} = [smooth_path ',1']; %#ok<AGROW>
    end

    spm_mat_path = fullfile(tmp_dir, 'SPM.mat');
    con_name = sprintf('Probe image category | %s | %s > 0', ...
        session_tag, strrep(contrast_tag, '_', ' '));

    ok = false;
    try
        matlabbatch = {};

        matlabbatch{1}.spm.stats.factorial_design.dir = {tmp_dir};
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
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = con_name;
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

        fprintf('Running SPM group batch in tmp dir: %s\n', tmp_dir);
        spm_jobman('run', matlabbatch);
        ok = true;
    catch ME
        fprintf('SPM error: %s\n', ME.message);
        ok = false;
    end

    if ~ok
        fprintf('[SKIP] Group SPM run failed for %s %s.\n', session_tag, contrast_tag);
        return;
    end

    spmT_path = fullfile(tmp_dir, 'spmT_0001.nii');
    if ~exist(spmT_path, 'file')
        fprintf('[SKIP] Expected spmT_0001.nii not found: %s\n', spmT_path);
        return;
    end

    copyfile(spmT_path, out_path_nii);
    gzip(out_path_nii);
    delete(out_path_nii);

    fprintf('Saved group t-map: %s\n', out_path_gz);

    if exist(tmp_dir, 'dir')
        try
            rmdir(tmp_dir, 's');
        catch
        end
    end
end
