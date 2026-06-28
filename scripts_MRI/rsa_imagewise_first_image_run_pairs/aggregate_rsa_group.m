% Group-level SPM analysis for first-image run-pair RSA searchlight maps.

%% ---------- Settings ----------
addpath('/Users/liuq13/Applications/spm12/');

base_dir = '/Users/liuq13/SPCD_data/rsa_maps';
analysis_subfolder = 'probe_imagewise_first_image_run_pairs';

sessions = {'premeal', 'postmeal'};
run_pairs = {'run1_vs_run2', 'run1_vs_run3', 'run2_vs_run3'};
models = {'odor_value', 'cue_type', 'image_identity'};
subset_tags = {'subset-all', 'subset-conditioned_only', 'subset-preconditioned_only'};

fwhm = 6;
mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';
skip_subs = {'sub-19', 'sub-33'};

%% ---------- Init SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

analysis_root = fullfile(base_dir, analysis_subfolder);
M = spm_read_vols(spm_vol(mask_file)) > 0;

%% ---------- Group analysis ----------
for rpi = 1:numel(run_pairs)
    run_pair = run_pairs{rpi};
    run_pair_dir = ['run_pair-' run_pair];

    for si = 1:numel(sessions)
        session_tag = sessions{si};
        input_dir = fullfile(analysis_root, session_tag, run_pair_dir);

        fprintf('\n============================================================\n');
        fprintf('>>> RUN-PAIR GROUP ANALYSIS | SESSION: %s | RUN PAIR: %s\n', ...
            session_tag, run_pair);
        fprintf('============================================================\n');

        if ~exist(input_dir, 'dir')
            fprintf('Input folder not found, skipping: %s\n', input_dir);
            continue;
        end

        for subseti = 1:numel(subset_tags)
            subset_tag = subset_tags{subseti};
            subset_suffix = ['_' subset_tag];

            for modeli = 1:numel(models)
                model_tag = models{modeli};
                pat = sprintf('sub-*_model-%s%s.nii.gz', model_tag, subset_suffix);
                all_gz = dir(fullfile(input_dir, pat));
                keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), all_gz);
                all_gz = all_gz(keep_idx);

                fprintf('Pattern: %s | Found %d maps.\n', pat, numel(all_gz));
                if isempty(all_gz)
                    continue;
                end

                spm_model_dir = fullfile(input_dir, 'group_spm', ...
                    sprintf('model-%s', model_tag), subset_tag);
                spm_input_dir = fullfile(spm_model_dir, 'spm_inputs');
                spm_mat_path = fullfile(spm_model_dir, 'SPM.mat');
                spmT_path = fullfile(spm_model_dir, 'spmT_0001.nii');

                if exist(spm_mat_path, 'file') || exist(spmT_path, 'file')
                    fprintf('SPM output already exists, skipping: %s\n', spm_model_dir);
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
                    Y(~M) = NaN;

                    [~, base_name, ~] = fileparts(nii_path);
                    masked_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
                    Vc = V;
                    Vc.fname = masked_path;
                    spm_write_vol(Vc, Y);

                    smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
                    spm_smooth(masked_path, smooth_path, [fwhm fwhm fwhm]);
                    smoothed_files{end+1, 1} = [smooth_path ',1']; %#ok<SAGROW>
                end

                con_name = sprintf('RSA %s | %s | %s | model-%s > 0', ...
                    session_tag, run_pair, subset_tag, model_tag);

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

                spm_jobman('run', matlabbatch);
                fprintf('Saved SPM group model: %s\n', spm_model_dir);
            end
        end
    end
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
