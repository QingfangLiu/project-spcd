% Group-level SPM analysis for pre/post modulation of cue-type RSA.
%
% This script tests whether cue-type representation differs between postmeal
% and premeal sessions. It writes both directions:
%
%   cue_type subset-all postmeal - cue_type subset-all premeal
%   cue_type subset-all premeal - cue_type subset-all postmeal
%
% It first creates paired within-subject difference maps, then runs persistent
% second-level SPM one-sample models against zero.

%% ---------- Settings ----------
addpath('/Users/liuq13/Applications/spm12/');

base_dir = '/Users/liuq13/SPCD_data/rsa_maps';
analysis_subfolder = 'probe_imagewise_first_image';

model_tag = 'cue_type';
subset_tag = 'subset-all';

contrast_tags = {'postmeal_minus_premeal', ...
                 'premeal_minus_postmeal'};
contrast_signs = [1, -1];

fwhm = 6; % smoothing kernel in mm

mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';
skip_subs = {'sub-19', 'sub-33'};

phase_diff_foldername = 'phase_diffs';
group_spm_foldername = 'group_spm';
subject_diff_foldername = 'subject_diffs';

analysis_root = fullfile(base_dir, analysis_subfolder);

pre_dir = fullfile(analysis_root, 'premeal');
post_dir = fullfile(analysis_root, 'postmeal');

phase_diff_root = fullfile(analysis_root, phase_diff_foldername);
subject_out_dir = fullfile(phase_diff_root, subject_diff_foldername);

if ~exist(phase_diff_root, 'dir')
    mkdir(phase_diff_root);
end

if ~exist(subject_out_dir, 'dir')
    mkdir(subject_out_dir);
end

%% ---------- Init SPM ----------
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

M = spm_read_vols(spm_vol(mask_file)) > 0;

%% ---------- Build paired subject differences ----------
fprintf('\n============================================================\n');
fprintf('>>> CUE-TYPE PHASE-DIFFERENCE GROUP ANALYSIS\n');
fprintf('============================================================\n');

if ~exist(pre_dir, 'dir')
    error('Premeal input folder not found: %s', pre_dir);
end

if ~exist(post_dir, 'dir')
    error('Postmeal input folder not found: %s', post_dir);
end

map_pat = sprintf('sub-*_model-%s_%s.nii.gz', model_tag, subset_tag);

pre_gz = dir(fullfile(pre_dir, map_pat));
post_gz = dir(fullfile(post_dir, map_pat));

pre_gz = drop_skipped_subjects(pre_gz, skip_subs);
post_gz = drop_skipped_subjects(post_gz, skip_subs);

fprintf('Pattern: %s\n', map_pat);
fprintf('Premeal files: %d | Postmeal files: %d\n', numel(pre_gz), numel(post_gz));

pre_map = files_by_subject(pre_gz);
post_map = files_by_subject(post_gz);

subj_keys = intersect(keys(pre_map), keys(post_map));
subj_keys = sort(subj_keys);

fprintf('Paired subjects: %d\n', numel(subj_keys));
if isempty(subj_keys)
    error('No paired premeal/postmeal cue_type maps found.');
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
        out_name = sprintf('%s_model-%s_%s_contrast-%s.nii.gz', ...
            sid, model_tag, subset_tag, contrast_tag);
        out_paths_gz{ci} = fullfile(subject_out_dir, out_name);
        subject_diff_gz{ci}{end+1, 1} = out_paths_gz{ci}; %#ok<SAGROW>

        if ~exist(out_paths_gz{ci}, 'file')
            needs_write = true;
        end
    end

    if ~needs_write
        fprintf('Subject phase differences exist, keeping: %s\n', sid);
        continue;
    end

    pre_nii = ensure_unzipped(pre_map(sid));
    post_nii = ensure_unzipped(post_map(sid));

    Vpre = spm_vol(pre_nii);
    Vpost = spm_vol(post_nii);

    assert_same_space(Vpre, Vpost, pre_nii, post_nii);

    Ypre = spm_read_vols(Vpre);
    Ypost = spm_read_vols(Vpost);

    Ybase = Ypost - Ypre;
    Ybase(~M) = NaN;

    for ci = 1:numel(contrast_tags)
        contrast_tag = contrast_tags{ci};
        out_path_gz = out_paths_gz{ci};

        if exist(out_path_gz, 'file')
            fprintf('Subject phase difference exists, keeping: %s\n', out_path_gz);
            continue;
        end

        Ydiff = contrast_signs(ci) .* Ybase;

        out_path_nii = erase(out_path_gz, '.gz');
        Vout = Vpost;
        Vout.fname = out_path_nii;
        Vout.descrip = sprintf('%s %s %s %s', ...
            sid, model_tag, subset_tag, contrast_tag);
        spm_write_vol(Vout, Ydiff);

        gzip(out_path_nii);
        delete(out_path_nii);

        fprintf('Saved subject phase difference: %s\n', out_path_gz);
    end
end

%% ---------- Group tests ----------
for ci = 1:numel(contrast_tags)
    contrast_tag = contrast_tags{ci};

    con_name = sprintf('RSA cue_type | subset-all | %s > 0', contrast_tag);
    spm_model_dir = fullfile(phase_diff_root, group_spm_foldername, ...
        sprintf('model-%s', model_tag), subset_tag, contrast_tag);

    run_one_sample_spm(subject_diff_gz{ci}, spm_model_dir, mask_file, ...
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

function run_one_sample_spm(input_gz_files, spm_model_dir, mask_file, M, fwhm, con_name)
    spm_input_dir = fullfile(spm_model_dir, 'spm_inputs');
    spm_mat_path = fullfile(spm_model_dir, 'SPM.mat');
    spmT_path = fullfile(spm_model_dir, 'spmT_0001.nii');

    if exist(spm_mat_path, 'file') || exist(spmT_path, 'file')
        fprintf('SPM model output already exists, skipping: %s\n', spm_model_dir);
        fprintf('Delete that model folder before rerunning this contrast.\n');
        return;
    end

    input_gz_files = input_gz_files(cellfun(@(p) exist(p, 'file') == 2, input_gz_files));

    if isempty(input_gz_files)
        fprintf('No input maps found for group test; skipping.\n');
        return;
    end

    if ~exist(spm_model_dir, 'dir')
        mkdir(spm_model_dir);
    end
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
        return;
    end

    ok = false;
    try
        matlabbatch = {};

        matlabbatch{1}.spm.stats.factorial_design.dir = {spm_model_dir};
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

        fprintf('Running persistent SPM batch: %s\n', spm_model_dir);
        spm_jobman('run', matlabbatch);

        ok = true;
    catch ME
        fprintf('SPM error: %s\n', ME.message);
        ok = false;
    end

    if ~ok
        fprintf('SPM run failed; skipping.\n');
        return;
    end

    if ~exist(spmT_path, 'file')
        fprintf('Expected spmT_0001.nii not found; skipping.\n');
        return;
    end

    fprintf('Saved SPM group model: %s\n', spm_model_dir);
    fprintf('Open this SPM.mat in SPM Results: %s\n', spm_mat_path);
end
