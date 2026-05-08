% Group-level SPM analysis for RSA searchlight maps (probe_imagewise layout),
% saving ONLY the group t-stat map (spmT_0001.nii) in a shallow folder layout.
%
% Revised version:
%   - Loops over the current primary RSA model: odor_value
%   - Runs per-session group maps for premeal, postmeal, and combined both.
%   - Loops over cue subset versions:
%       1) subset-all
%       2) subset-conditioned_only
%       3) subset-preconditioned_only
%   - Uses maps generated with the default second_image_only button scope.
%   - No new helper functions added.
%   - SPM batch code is kept inline.

%% ---------- Settings ----------
addpath('/Users/liuq13/Applications/spm12/');

base_dir = '/Users/liuq13/SPCD_data/rsa_maps';
analysis_subfolder = 'probe_imagewise';

sessions = {'premeal', 'postmeal', 'both'};
inputs   = {'input-beta'}; %#ok<NASGU>

models = {'odor_value'};

% cue subset versions
%   subset-all = all cue maps
%   subset-conditioned_only = conditioned cue maps
%   subset-preconditioned_only = preconditioned cue maps
subset_tags = {'subset-all', 'subset-conditioned_only', 'subset-preconditioned_only'};

fwhm = 6; % smoothing kernel in mm

mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';
skip_subs = {'sub-19', 'sub-33'};

collapsed_tag = 'collapsed';
group_out_foldername = 'group_tmaps';

tmp_root = fullfile(base_dir, analysis_subfolder, '_tmp_spm_work');
if ~exist(tmp_root, 'dir')
    mkdir(tmp_root);
end

%% ---------- Init SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

analysis_root = fullfile(base_dir, analysis_subfolder);

M = spm_read_vols(spm_vol(mask_file)) > 0;

%% ---------- Part 1: Per-session group analysis ----------
for si = 1:numel(sessions)
    session_tag = sessions{si};

    fprintf('\n============================================================\n');
    fprintf('>>> PER-SESSION GROUP ANALYSIS | SESSION: %s\n', session_tag);
    fprintf('============================================================\n');

    input_dir = fullfile(analysis_root, session_tag);
    if ~exist(input_dir, 'dir')
        fprintf('Input folder not found, skipping: %s\n', input_dir);
        continue;
    end

    out_dir_session_input = fullfile(analysis_root, session_tag, group_out_foldername);

    for subseti = 1:numel(subset_tags)
        subset_tag = subset_tags{subseti};

        subset_suffix = ['_' subset_tag];
        subset_label = subset_tag;

        fprintf('\n--- Subset version: %s ---\n', subset_label);

        for modeli = 1:numel(models)
            model_tag = models{modeli};

            pat = sprintf('sub-*_model-%s%s.nii.gz', ...
                model_tag, subset_suffix);

            all_gz = dir(fullfile(input_dir, pat));

            keep_idx = ~arrayfun(@(f) any(contains(f.name, skip_subs)), all_gz);
            all_gz = all_gz(keep_idx);

            fprintf('Pattern: %s\n', pat);
            fprintf('Found %d .nii.gz maps.\n', numel(all_gz));

            if isempty(all_gz)
                continue;
            end

            out_name = sprintf('group_model-%s%s.nii.gz', ...
                model_tag, subset_suffix);

            out_path_gz = fullfile(out_dir_session_input, out_name);

            if exist(out_path_gz, 'file')
                fprintf('Output exists, skipping: %s\n', out_path_gz);
                continue;
            end

            tmp_dir = tempname(tmp_root);
            if ~exist(tmp_dir, 'dir')
                mkdir(tmp_dir);
            end

            spm_input_dir = fullfile(tmp_dir, 'spm_inputs');
            if ~exist(spm_input_dir, 'dir')
                mkdir(spm_input_dir);
            end

            smoothed_files = cell(0,1);

            for fi = 1:numel(all_gz)
                gz_path = fullfile(all_gz(fi).folder, all_gz(fi).name);

                [gz_folder, gz_name, gz_ext] = fileparts(gz_path);
                if ~strcmp(gz_ext, '.gz')
                    error('Expected .nii.gz file, got: %s', gz_path);
                end

                nii_path = fullfile(gz_folder, gz_name);
                if ~exist(nii_path, 'file')
                    gunzip(gz_path, gz_folder);
                end

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

                smoothed_files{end+1,1} = [smooth_path ',1']; %#ok<SAGROW>
            end

            if isempty(smoothed_files)
                fprintf('No usable smoothed files produced; skipping.\n');
                if exist(tmp_dir, 'dir')
                    try
                        rmdir(tmp_dir, 's');
                    catch
                    end
                end
                continue;
            end

            spm_mat_path = fullfile(tmp_dir, 'SPM.mat');

            con_name = sprintf('RSA %s | subset-%s | model-%s > 0', ...
                session_tag, subset_label, model_tag);

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

                matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = con_name;
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
                fprintf('SPM run failed; skipping.\n');
                if exist(tmp_dir, 'dir')
                    try
                        rmdir(tmp_dir, 's');
                    catch
                    end
                end
                continue;
            end

            spmT_path = fullfile(tmp_dir, 'spmT_0001.nii');
            if ~exist(spmT_path, 'file')
                fprintf('Expected spmT_0001.nii not found; skipping.\n');
                if exist(tmp_dir, 'dir')
                    try
                        rmdir(tmp_dir, 's');
                    catch
                    end
                end
                continue;
            end

            if ~exist(out_dir_session_input, 'dir')
                mkdir(out_dir_session_input);
            end

            out_path_nii = fullfile(out_dir_session_input, erase(out_name, '.gz'));
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
    end
end

%% ---------- Part 2: Collapsed across sessions ----------
fprintf('\n============================================================\n');
fprintf('>>> COLLAPSED (WITHIN-SUBJECT AVERAGE) GROUP ANALYSIS\n');
fprintf('============================================================\n');

out_dir_collapsed_input = fullfile(analysis_root, collapsed_tag, group_out_foldername);

pre_dir  = fullfile(analysis_root, 'premeal');
post_dir = fullfile(analysis_root, 'postmeal');

for subseti = 1:numel(subset_tags)
    subset_tag = subset_tags{subseti};

    subset_suffix = ['_' subset_tag];
    subset_label = subset_tag;

    fprintf('\n--- Subset version: %s ---\n', subset_label);

    for modeli = 1:numel(models)
        model_tag = models{modeli};

        pat = sprintf('sub-*_model-%s%s.nii.gz', ...
            model_tag, subset_suffix);

        pre_gz  = dir(fullfile(pre_dir, pat));
        post_gz = dir(fullfile(post_dir, pat));

        pre_keep  = ~arrayfun(@(f) any(contains(f.name, skip_subs)), pre_gz);
        post_keep = ~arrayfun(@(f) any(contains(f.name, skip_subs)), post_gz);
        pre_gz  = pre_gz(pre_keep);
        post_gz = post_gz(post_keep);

        fprintf('Pattern: %s\n', pat);
        fprintf('Premeal files: %d | Postmeal files: %d\n', numel(pre_gz), numel(post_gz));

        if isempty(pre_gz) || isempty(post_gz)
            fprintf('Not enough files to collapse; skipping.\n');
            continue;
        end

        pre_map  = containers.Map();
        post_map = containers.Map();

        for k = 1:numel(pre_gz)
            nm = pre_gz(k).name;
            tok = regexp(nm, '^(sub-\d+)_', 'tokens', 'once');
            if isempty(tok)
                continue;
            end
            pre_map(tok{1}) = fullfile(pre_gz(k).folder, pre_gz(k).name);
        end

        for k = 1:numel(post_gz)
            nm = post_gz(k).name;
            tok = regexp(nm, '^(sub-\d+)_', 'tokens', 'once');
            if isempty(tok)
                continue;
            end
            post_map(tok{1}) = fullfile(post_gz(k).folder, post_gz(k).name);
        end

        pre_keys = pre_map.keys;
        subj_keys = {};
        for k = 1:numel(pre_keys)
            s = pre_keys{k};
            if isKey(post_map, s)
                subj_keys{end+1} = s; %#ok<AGROW>
            end
        end
        subj_keys = sort(subj_keys);

        fprintf('Subjects with both sessions: %d\n', numel(subj_keys));
        if isempty(subj_keys)
            fprintf('No overlapping subjects; skipping.\n');
            continue;
        end

        out_name = sprintf('group_model-%s%s.nii.gz', ...
            model_tag, subset_suffix);

        out_path_gz = fullfile(out_dir_collapsed_input, out_name);

        if exist(out_path_gz, 'file')
            fprintf('Output exists, skipping: %s\n', out_path_gz);
            continue;
        end

        tmp_dir = tempname(tmp_root);
        if ~exist(tmp_dir, 'dir')
            mkdir(tmp_dir);
        end

        spm_input_dir = fullfile(tmp_dir, 'spm_inputs');
        if ~exist(spm_input_dir, 'dir')
            mkdir(spm_input_dir);
        end

        smoothed_files = cell(0,1);

        for k = 1:numel(subj_keys)
            sid = subj_keys{k};

            pre_gz_path = pre_map(sid);
            post_gz_path = post_map(sid);

            [pre_folder, pre_name, pre_ext] = fileparts(pre_gz_path);
            if ~strcmp(pre_ext, '.gz')
                error('Expected .nii.gz file, got: %s', pre_gz_path);
            end
            pre_nii = fullfile(pre_folder, pre_name);
            if ~exist(pre_nii, 'file')
                gunzip(pre_gz_path, pre_folder);
            end

            [post_folder, post_name, post_ext] = fileparts(post_gz_path);
            if ~strcmp(post_ext, '.gz')
                error('Expected .nii.gz file, got: %s', post_gz_path);
            end
            post_nii = fullfile(post_folder, post_name);
            if ~exist(post_nii, 'file')
                gunzip(post_gz_path, post_folder);
            end

            Vpre  = spm_vol(pre_nii);
            Vpost = spm_vol(post_nii);

            Ypre  = spm_read_vols(Vpre);
            Ypost = spm_read_vols(Vpost);

            Ypre(~M)  = NaN;
            Ypost(~M) = NaN;

            Yavg = (Ypre + Ypost) ./ 2;

            base_name = sprintf('%s_collapsed_model-%s%s', ...
                sid, model_tag, subset_suffix);

            masked_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
            Vc = Vpre;
            Vc.fname = masked_path;
            spm_write_vol(Vc, Yavg);

            smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
            fprintf('  Collapsed avg + smooth: %s\n', base_name);
            spm_smooth(masked_path, smooth_path, [fwhm fwhm fwhm]);

            smoothed_files{end+1,1} = [smooth_path ',1']; %#ok<SAGROW>
        end

        if isempty(smoothed_files)
            fprintf('No usable collapsed smoothed files; skipping.\n');
            if exist(tmp_dir, 'dir')
                try
                    rmdir(tmp_dir, 's');
                catch
                end
            end
            continue;
        end

        spm_mat_path = fullfile(tmp_dir, 'SPM.mat');

        con_name = sprintf('RSA collapsed | subset-%s | model-%s > 0', ...
            subset_label, model_tag);

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

            matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = con_name;
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
            fprintf('SPM run failed; skipping.\n');
            if exist(tmp_dir, 'dir')
                try
                    rmdir(tmp_dir, 's');
                catch
                end
            end
            continue;
        end

        spmT_path = fullfile(tmp_dir, 'spmT_0001.nii');
        if ~exist(spmT_path, 'file')
            fprintf('Expected spmT_0001.nii not found; skipping.\n');
            if exist(tmp_dir, 'dir')
                try
                    rmdir(tmp_dir, 's');
                catch
                end
            end
            continue;
        end

        if ~exist(out_dir_collapsed_input, 'dir')
            mkdir(out_dir_collapsed_input);
        end

        out_path_nii = fullfile(out_dir_collapsed_input, erase(out_name, '.gz'));
        copyfile(spmT_path, out_path_nii);

        gzip(out_path_nii);
        delete(out_path_nii);

        fprintf('Saved collapsed group t-map: %s\n', out_path_gz);

        if exist(tmp_dir, 'dir')
            try
                rmdir(tmp_dir, 's');
            catch
            end
        end
    end
end

fprintf('\nAll done.\n');
