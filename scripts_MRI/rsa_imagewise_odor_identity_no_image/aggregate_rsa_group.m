% Group-level SPM analysis for premeal odor-identity no-image RSA searchlight maps.
%
% This variant aggregates only:
%   - model-odor_identity_no_image / subset-conditioned_odor_only
%   - model-odor_identity_no_image / subset-preconditioned_odor_only

%% ---------- Settings ----------
addpath('/Users/liuq13/Applications/spm12/');

base_dir = '/Users/liuq13/SPCD_data/rsa_maps';
analysis_subfolder = 'probe_imagewise_odor_identity_no_image';
session_tag = 'premeal';

model_tag = 'odor_identity_no_image';
subset_tags = {'subset-conditioned_odor_only', ...
               'subset-preconditioned_odor_only'};

fwhm = 6; % smoothing kernel in mm

mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';
skip_subs = {'sub-19', 'sub-33'};

%% ---------- Init SPM ----------
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

analysis_root = fullfile(base_dir, analysis_subfolder, session_tag);

if ~exist(analysis_root, 'dir')
    error('Input folder not found: %s', analysis_root);
end

M = spm_read_vols(spm_vol(mask_file)) > 0;

%% ---------- Group analysis ----------
fprintf('\n============================================================\n');
fprintf('>>> PREMEAL ODOR-IDENTITY NO-IMAGE RSA GROUP ANALYSIS\n');
fprintf('============================================================\n');

for subseti = 1:numel(subset_tags)
    subset_tag = subset_tags{subseti};
    subset_suffix = ['_' subset_tag];

    fprintf('\n--- Subset version: %s ---\n', subset_tag);

    pat = sprintf('sub-*_model-%s%s.nii.gz', model_tag, subset_suffix);
    all_gz = dir(fullfile(analysis_root, pat));

    keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), all_gz);
    all_gz = all_gz(keep_idx);

    fprintf('Pattern: %s\n', pat);
    fprintf('Found %d .nii.gz maps.\n', numel(all_gz));

    if isempty(all_gz)
        continue;
    end

    spm_model_dir = fullfile(analysis_root, 'group_spm', ...
        sprintf('model-%s', model_tag), subset_tag);
    spm_input_dir = fullfile(spm_model_dir, 'spm_inputs');
    spm_mat_path = fullfile(spm_model_dir, 'SPM.mat');
    spmT_path = fullfile(spm_model_dir, 'spmT_0001.nii');

    if exist(spm_mat_path, 'file') || exist(spmT_path, 'file')
        fprintf('SPM model output already exists, skipping: %s\n', spm_model_dir);
        fprintf('Delete that model folder before rerunning this contrast.\n');
        continue;
    end

    if ~exist(spm_model_dir, 'dir')
        mkdir(spm_model_dir);
    end
    if ~exist(spm_input_dir, 'dir')
        mkdir(spm_input_dir);
    end

    smoothed_files = cell(0, 1);

    for fi = 1:numel(all_gz)
        gz_path = fullfile(all_gz(fi).folder, all_gz(fi).name);
        nii_path = ensure_unzipped(gz_path);

        V = spm_vol(nii_path);
        Y = spm_read_vols(V);

        [~, base_name, ~] = fileparts(nii_path);
        fprintf('  Mask->NaN + smooth: %s\n', base_name);

        Y(~M) = NaN;

        masked_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
        Vc = V;
        Vc.fname = masked_path;
        spm_write_vol(Vc, Y);

        smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
        spm_smooth(masked_path, smooth_path, [fwhm fwhm fwhm]);

        smoothed_files{end+1, 1} = [smooth_path ',1']; %#ok<SAGROW>
    end

    if isempty(smoothed_files)
        fprintf('No usable smoothed files produced; skipping.\n');
        continue;
    end

    con_name = sprintf('RSA premeal | %s | model-%s > 0', ...
        subset_tag, model_tag);

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
        continue;
    end

    if ~exist(spmT_path, 'file')
        fprintf('Expected spmT_0001.nii not found; skipping.\n');
        continue;
    end

    fprintf('Saved SPM group model: %s\n', spm_model_dir);
    fprintf('Open this SPM.mat in SPM Results: %s\n', spm_mat_path);
end

fprintf('\nAll done.\n');

%% ---------- Helper functions ----------
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
