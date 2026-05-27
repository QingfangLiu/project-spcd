% Group-level SPM analysis for cue-type modulation of odor-value RSA.
%
% This script tests whether odor-value representation differs between
% conditioned and preconditioned cue subsets. It writes both directions:
%
%   odor_value subset-conditioned_only - odor_value subset-preconditioned_only
%   odor_value subset-preconditioned_only - odor_value subset-conditioned_only
%
% It first creates paired within-subject difference maps, then runs one-sample
% SPM t-tests against zero for premeal, postmeal, both, and collapsed pre/post.

%% ---------- Settings ----------
addpath('/Users/liuq13/Applications/spm12/');

base_dir = '/Users/liuq13/SPCD_data/rsa_maps';
analysis_subfolder = 'probe_imagewise';

sessions = {'premeal', 'postmeal', 'both'};

model_tag = 'odor_value';
cond_subset = 'subset-conditioned_only';
pre_subset = 'subset-preconditioned_only';
contrast_tags = {'conditioned_minus_preconditioned', ...
                 'preconditioned_minus_conditioned'};
contrast_signs = [1, -1];

fwhm = 6; % smoothing kernel in mm

mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';
skip_subs = {'sub-19', 'sub-33'};

collapsed_tag = 'collapsed';
group_out_foldername = 'group_tmaps';
subject_diff_foldername = 'subject_diffs';

analysis_root = fullfile(base_dir, analysis_subfolder);
tmp_root = fullfile(analysis_root, '_tmp_spm_work');
if ~exist(tmp_root, 'dir')
    mkdir(tmp_root);
end

%% ---------- Init SPM ----------
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

M = spm_read_vols(spm_vol(mask_file)) > 0;

%% ---------- Part 1: Per-session paired subset differences ----------
for si = 1:numel(sessions)
    session_tag = sessions{si};

    fprintf('\n============================================================\n');
    fprintf('>>> SUBSET-DIFFERENCE GROUP ANALYSIS | SESSION: %s\n', session_tag);
    fprintf('============================================================\n');

    input_dir = fullfile(analysis_root, session_tag);
    if ~exist(input_dir, 'dir')
        fprintf('Input folder not found, skipping: %s\n', input_dir);
        continue;
    end

    cond_pat = sprintf('sub-*_model-%s_%s.nii.gz', model_tag, cond_subset);
    pre_pat = sprintf('sub-*_model-%s_%s.nii.gz', model_tag, pre_subset);

    cond_gz = dir(fullfile(input_dir, cond_pat));
    pre_gz = dir(fullfile(input_dir, pre_pat));

    cond_gz = drop_skipped_subjects(cond_gz, skip_subs);
    pre_gz = drop_skipped_subjects(pre_gz, skip_subs);

    fprintf('Conditioned pattern: %s\n', cond_pat);
    fprintf('Preconditioned pattern: %s\n', pre_pat);
    fprintf('Conditioned files: %d | Preconditioned files: %d\n', ...
        numel(cond_gz), numel(pre_gz));

    cond_map = files_by_subject(cond_gz);
    pre_map = files_by_subject(pre_gz);

    subj_keys = intersect(keys(cond_map), keys(pre_map));
    subj_keys = sort(subj_keys);

    fprintf('Paired subjects: %d\n', numel(subj_keys));
    if isempty(subj_keys)
        fprintf('No paired subject maps found; skipping.\n');
        continue;
    end

    subject_out_dir = fullfile(analysis_root, session_tag, subject_diff_foldername);
    if ~exist(subject_out_dir, 'dir')
        mkdir(subject_out_dir);
    end

    subject_diff_gz = cell(numel(contrast_tags), 1);
    for ci = 1:numel(contrast_tags)
        subject_diff_gz{ci} = cell(0, 1);
    end

    for k = 1:numel(subj_keys)
        sid = subj_keys{k};

        out_paths_gz = cell(numel(contrast_tags), 1);
        needs_write = false;

        for ci = 1:numel(contrast_tags)
            contrast_tag = contrast_tags{ci};
            out_name = sprintf('%s_model-%s_contrast-%s.nii.gz', ...
                sid, model_tag, contrast_tag);
            out_paths_gz{ci} = fullfile(subject_out_dir, out_name);
            subject_diff_gz{ci}{end+1, 1} = out_paths_gz{ci}; %#ok<SAGROW>

            if ~exist(out_paths_gz{ci}, 'file')
                needs_write = true;
            end
        end

        if ~needs_write
            fprintf('Subject differences exist, keeping: %s\n', sid);
            continue;
        end

        cond_nii = ensure_unzipped(cond_map(sid));
        pre_nii = ensure_unzipped(pre_map(sid));

        Vcond = spm_vol(cond_nii);
        Vpre = spm_vol(pre_nii);

        assert_same_space(Vcond, Vpre, cond_nii, pre_nii);

        Ycond = spm_read_vols(Vcond);
        Ypre = spm_read_vols(Vpre);

        Ybase = Ycond - Ypre;
        Ybase(~M) = NaN;

        for ci = 1:numel(contrast_tags)
            contrast_tag = contrast_tags{ci};
            out_path_gz = out_paths_gz{ci};

            if exist(out_path_gz, 'file')
                fprintf('Subject difference exists, keeping: %s\n', out_path_gz);
                continue;
            end

            Ydiff = contrast_signs(ci) .* Ybase;

            out_path_nii = erase(out_path_gz, '.gz');
            Vout = Vcond;
            Vout.fname = out_path_nii;
            Vout.descrip = sprintf('%s %s %s', sid, model_tag, contrast_tag);
            spm_write_vol(Vout, Ydiff);

            gzip(out_path_nii);
            delete(out_path_nii);

            fprintf('Saved subject difference: %s\n', out_path_gz);
        end
    end

    for ci = 1:numel(contrast_tags)
        contrast_tag = contrast_tags{ci};

        out_group_gz = fullfile(analysis_root, session_tag, group_out_foldername, ...
            sprintf('group_model-%s_contrast-%s.nii.gz', model_tag, contrast_tag));

        con_name = sprintf('RSA %s | model-%s | %s > 0', ...
            session_tag, model_tag, contrast_tag);

        run_one_sample_spm(subject_diff_gz{ci}, out_group_gz, tmp_root, mask_file, ...
            M, fwhm, con_name);
    end
end

%% ---------- Part 2: Collapsed across premeal and postmeal ----------
fprintf('\n============================================================\n');
fprintf('>>> COLLAPSED SUBSET-DIFFERENCE GROUP ANALYSIS\n');
fprintf('============================================================\n');

pre_diff_dir = fullfile(analysis_root, 'premeal', subject_diff_foldername);
post_diff_dir = fullfile(analysis_root, 'postmeal', subject_diff_foldername);

for ci = 1:numel(contrast_tags)
    contrast_tag = contrast_tags{ci};

    fprintf('\n--- Contrast: %s ---\n', contrast_tag);

    diff_pat = sprintf('sub-*_model-%s_contrast-%s.nii.gz', model_tag, contrast_tag);

    pre_gz = dir(fullfile(pre_diff_dir, diff_pat));
    post_gz = dir(fullfile(post_diff_dir, diff_pat));

    pre_gz = drop_skipped_subjects(pre_gz, skip_subs);
    post_gz = drop_skipped_subjects(post_gz, skip_subs);

    fprintf('Premeal difference files: %d | Postmeal difference files: %d\n', ...
        numel(pre_gz), numel(post_gz));

    pre_map = files_by_subject(pre_gz);
    post_map = files_by_subject(post_gz);

    subj_keys = intersect(keys(pre_map), keys(post_map));
    subj_keys = sort(subj_keys);

    fprintf('Subjects with both premeal and postmeal differences: %d\n', numel(subj_keys));

    if isempty(subj_keys)
        fprintf('No paired pre/post subject differences found; skipping collapsed analysis.\n');
        continue;
    end

    collapsed_subject_out_dir = fullfile(analysis_root, collapsed_tag, subject_diff_foldername);
    if ~exist(collapsed_subject_out_dir, 'dir')
        mkdir(collapsed_subject_out_dir);
    end

    collapsed_diff_gz = cell(0, 1);

    for k = 1:numel(subj_keys)
        sid = subj_keys{k};

        out_name = sprintf('%s_model-%s_contrast-%s.nii.gz', ...
            sid, model_tag, contrast_tag);
        out_path_gz = fullfile(collapsed_subject_out_dir, out_name);
        collapsed_diff_gz{end+1, 1} = out_path_gz; %#ok<SAGROW>

        if exist(out_path_gz, 'file')
            fprintf('Collapsed subject difference exists, keeping: %s\n', out_path_gz);
            continue;
        end

        pre_nii = ensure_unzipped(pre_map(sid));
        post_nii = ensure_unzipped(post_map(sid));

        Vpre = spm_vol(pre_nii);
        Vpost = spm_vol(post_nii);

        assert_same_space(Vpre, Vpost, pre_nii, post_nii);

        Ypre = spm_read_vols(Vpre);
        Ypost = spm_read_vols(Vpost);

        Yavg = (Ypre + Ypost) ./ 2;
        Yavg(~M) = NaN;

        out_path_nii = erase(out_path_gz, '.gz');
        Vout = Vpre;
        Vout.fname = out_path_nii;
        Vout.descrip = sprintf('%s collapsed %s %s', sid, model_tag, contrast_tag);
        spm_write_vol(Vout, Yavg);

        gzip(out_path_nii);
        delete(out_path_nii);

        fprintf('Saved collapsed subject difference: %s\n', out_path_gz);
    end

    out_group_gz = fullfile(analysis_root, collapsed_tag, group_out_foldername, ...
        sprintf('group_model-%s_contrast-%s.nii.gz', model_tag, contrast_tag));

    con_name = sprintf('RSA collapsed | model-%s | %s > 0', ...
        model_tag, contrast_tag);

    run_one_sample_spm(collapsed_diff_gz, out_group_gz, tmp_root, mask_file, ...
        M, fwhm, con_name);
end

fprintf('\nAll done.\n');

%% ---------- Helper functions ----------
function files = drop_skipped_subjects(files, skip_subs)
    keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), files);
    files = files(keep_idx);
end

function m = files_by_subject(files)
    m = containers.Map();

    for k = 1:numel(files)
        nm = files(k).name;
        tok = regexp(nm, '^(sub-\d+)_', 'tokens', 'once');
        if isempty(tok)
            continue;
        end
        m(tok{1}) = fullfile(files(k).folder, files(k).name);
    end
end

function nii_path = ensure_unzipped(gz_path)
    [folder, name, ext] = fileparts(gz_path);

    if strcmp(ext, '.gz')
        nii_path = fullfile(folder, name);
        if exist(nii_path, 'file')
            gz_info = dir(gz_path);
            nii_info = dir(nii_path);
            if ~isempty(gz_info) && ~isempty(nii_info) && gz_info.datenum > nii_info.datenum
                delete(nii_path);
            end
        end
        if ~exist(nii_path, 'file')
            gunzip(gz_path, folder);
        end
    else
        nii_path = gz_path;
    end
end

function assert_same_space(V1, V2, path1, path2)
    if ~isequal(V1.dim, V2.dim) || max(abs(V1.mat(:) - V2.mat(:))) > 1e-6
        error('NIfTI spaces do not match:\n%s\n%s', path1, path2);
    end
end

function run_one_sample_spm(input_gz_files, out_path_gz, tmp_root, mask_file, M, fwhm, con_name)
    if exist(out_path_gz, 'file')
        fprintf('Output exists, skipping: %s\n', out_path_gz);
        return;
    end

    input_gz_files = input_gz_files(cellfun(@(p) exist(p, 'file') == 2, input_gz_files));

    if isempty(input_gz_files)
        fprintf('No input maps found for group test; skipping.\n');
        return;
    end

    out_dir = fileparts(out_path_gz);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    tmp_dir = tempname(tmp_root);
    if ~exist(tmp_dir, 'dir')
        mkdir(tmp_dir);
    end

    spm_input_dir = fullfile(tmp_dir, 'spm_inputs');
    if ~exist(spm_input_dir, 'dir')
        mkdir(spm_input_dir);
    end

    smoothed_files = cell(0, 1);

    for fi = 1:numel(input_gz_files)
        nii_path = ensure_unzipped(input_gz_files{fi});

        V = spm_vol(nii_path);
        Y = spm_read_vols(V);
        Y(~M) = NaN;

        [~, base_name, ~] = fileparts(nii_path);

        masked_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
        smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);

        Vc = V;
        Vc.fname = masked_path;

        spm_write_vol(Vc, Y);
        spm_smooth(masked_path, smooth_path, [fwhm fwhm fwhm]);

        smoothed_files{end+1, 1} = [smooth_path ',1']; %#ok<AGROW>

        fprintf('  Mask->NaN + smooth: %s\n', base_name);
    end

    if isempty(smoothed_files)
        fprintf('No usable smoothed files produced; skipping.\n');
        cleanup_tmp_dir(tmp_dir);
        return;
    end

    spm_mat_path = fullfile(tmp_dir, 'SPM.mat');

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

        fprintf('Running SPM batch in tmp work dir: %s\n', tmp_dir);
        spm_jobman('run', matlabbatch);

        ok = true;
    catch ME
        fprintf('SPM error: %s\n', ME.message);
        ok = false;
    end

    if ~ok
        fprintf('SPM run failed; skipping save.\n');
        cleanup_tmp_dir(tmp_dir);
        return;
    end

    spmT_path = fullfile(tmp_dir, 'spmT_0001.nii');
    if ~exist(spmT_path, 'file')
        fprintf('Expected spmT_0001.nii not found; skipping save.\n');
        cleanup_tmp_dir(tmp_dir);
        return;
    end

    out_path_nii = erase(out_path_gz, '.gz');
    copyfile(spmT_path, out_path_nii);

    gzip(out_path_nii);
    delete(out_path_nii);

    fprintf('Saved group t-map: %s\n', out_path_gz);

    cleanup_tmp_dir(tmp_dir);
end

function cleanup_tmp_dir(tmp_dir)
    if exist(tmp_dir, 'dir')
        try
            rmdir(tmp_dir, 's');
        catch
        end
    end
end
