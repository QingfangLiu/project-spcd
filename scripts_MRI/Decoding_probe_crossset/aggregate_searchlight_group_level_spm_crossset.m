% aggregate_searchlight_group_level_spm_crossset.m
%
% SPM12 version of aggregate_searchlight_group_level.py (probe_crossset)
% - One-sample t-test in SPM on (accuracy - chance)
% - Explicit GM mask: base_dir/masks/gm_0.1.nii
% - Keeps results separate in .../group_stats_spm/session-<tag>/

%% ---------- Settings ----------
base_dir   = '/Users/liuq13/SPCD_data/searchlight_maps';
subfolder  = 'probe_crossset';

modes      = {'strict', 'set2set'};
cues       = {'A', 'B'};
class_list = [2, 3];

session_tag = 'premeal';      % matches the Python pattern
radius_str  = '8mm';
fwhm        = 6;              % mm FWHM
group_stats_suffix = 'group_stats_spm';

% explicit GM mask
mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';
if ~exist(mask_file, 'file')
    error('Explicit mask not found: %s', mask_file);
end

%% ---------- Init SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

searchlight_dir = fullfile(base_dir, subfolder);
chance_fun = @(n_cls) (n_cls==2)*0.5 + (n_cls==3)*(1/3);

%% ---------- Main loops ----------
for n_cls = class_list
    for mi = 1:numel(modes)
        mode = modes{mi};
        for ci = 1:numel(cues)
            cue = cues{ci};

            fprintf('\n>>> SPM group analysis | classes=%d | mode=%s | cue=%s | session=%s\n', ...
                n_cls, mode, cue, session_tag);

            combo_dir  = fullfile(searchlight_dir, ...
                sprintf('mode-%s_cue-%s_classes-%d', mode, cue, n_cls));
            output_dir = fullfile(combo_dir, group_stats_suffix, ...
                sprintf('session-%s', session_tag));
            if ~exist(output_dir, 'dir'), mkdir(output_dir); end

            spm_input_dir = fullfile(output_dir, 'spm_inputs');
            if ~exist(spm_input_dir, 'dir'), mkdir(spm_input_dir); end

            %% ---- Find subject maps (.nii.gz) ----
            pattern = fullfile(searchlight_dir, ...
                sprintf('sub-*_session-%s_mode-%s_cue-%s_classes-%d_radius-%s.nii.gz', ...
                        session_tag, mode, cue, n_cls, radius_str));

            map_files_gz = dir(pattern);
            fprintf('Found %d searchlight maps.\n', numel(map_files_gz));
            if isempty(map_files_gz)
                fprintf('No maps found. Skipping.\n');
                continue;
            end

            %% ---- Gunzip, center at chance, smooth ----
            chance_val     = chance_fun(n_cls);
            smoothed_files = {};

            for i = 1:numel(map_files_gz)
                gz_path = fullfile(map_files_gz(i).folder, map_files_gz(i).name);

                % decompose 'xxx.nii.gz' -> 'xxx.nii'
                [gz_folder, gz_name, gz_ext] = fileparts(gz_path); % gz_ext = '.gz'
                if ~strcmp(gz_ext, '.gz')
                    error('Expected .nii.gz file, got: %s', gz_path);
                end
                nii_path = fullfile(gz_folder, gz_name);  % this has .nii inside gz_name

                if ~exist(nii_path, 'file')
                    fprintf('  Gunzip: %s\n', gz_path);
                    gunzip(gz_path, gz_folder);
                end

                % Load accuracy map
                V = spm_vol(nii_path);
                Y = spm_read_vols(V);

                % Load the GM mask 
                M = spm_read_vols(spm_vol(mask_file)) > 0;

                % Set outside-brain voxels to NaN
                Y(~M) = NaN;

                % Center at chance: acc - chance
                Yc = Y - chance_val;

                % Save centered image
                [~, base_name, ~] = fileparts(nii_path);  % strip '.nii'
                centered_path = fullfile(spm_input_dir, ['c_' base_name '.nii']);
                Vc       = V;
                Vc.fname = centered_path;
                spm_write_vol(Vc, Yc);

                % Smooth
                smooth_path = fullfile(spm_input_dir, ['s_' base_name '.nii']);
                fprintf('  Center + smooth: %s -> %s\n', centered_path, smooth_path);
                spm_smooth(centered_path, smooth_path, [fwhm fwhm fwhm]);

                smoothed_files{end+1,1} = [smooth_path ',1']; 
            end

            %% ---- SPM one-sample t-test on (acc - chance) ----
            matlabbatch = {};

            % Factorial design
            matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir};
            matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = smoothed_files;

            % No covariates
            matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});

            % Explicit mask, no implicit mask
            matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
            matlabbatch{1}.spm.stats.factorial_design.masking.im = 0;
            matlabbatch{1}.spm.stats.factorial_design.masking.em = {mask_file};

            % Global options (defaults)
            matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
            matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
            matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

            % Model estimation
            spm_mat_path = fullfile(output_dir, 'SPM.mat');
            matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat_path};
            matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
            matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

            % Contrast: Decoding > chance
            matlabbatch{3}.spm.stats.con.spmmat = {spm_mat_path};
            matlabbatch{3}.spm.stats.con.delete = 0;
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = 'Decoding > chance';
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

            % Run batch
            fprintf('Running SPM batch in: %s\n', output_dir);
            spm_jobman('run', matlabbatch);

            fprintf('  -> Done. T-map: %s\n', fullfile(output_dir, 'spmT_0001.nii'));
        end
    end
end

fprintf('\nAll combinations finished.\n');
