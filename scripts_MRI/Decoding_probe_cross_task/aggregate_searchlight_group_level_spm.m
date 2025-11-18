% aggregate_searchlight_group_level_spm.m
% MATLAB/SPM12 version of your Python aggregation for searchlight maps.

addpath('/Users/liuq13/Applications/spm12/')

%% ---------- Settings ----------
base_dir   = '/Users/liuq13/SPCD_data/searchlight_maps';
subfolder  = 'probe_crosstask';

mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';

% Check mask before running
if ~exist(mask_file, 'file')
    error('Mask file not found: %s', mask_file);
end
fprintf('Using mask: %s\n', mask_file);

cue_types  = {'A','B','AB'};
class_list = [2,3];

session_tag = 'premeal';
radius_str  = '8mm';
fwhm        = 6;
mode_tag    = 'crosstask';

group_stats_suffix = 'group_stats_spm';

%% ---------- Initialize SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

searchlight_dir = fullfile(base_dir, subfolder);
chance_fun = @(n_cls) (n_cls==2)*0.5 + (n_cls==3)*(1/3);

%% ---------- Main loops ----------
for n_cls = class_list
    for ci = 1:numel(cue_types)
        cue = cue_types{ci};

        fprintf('\n>>> SPM group analysis | classes=%d | cue=%s | session=%s\n', ...
            n_cls, cue, session_tag);

        combo_dir = fullfile(searchlight_dir, ...
            sprintf('mode-%s_cue-%s_classes-%d', mode_tag, cue, n_cls));

        output_dir = fullfile(combo_dir, group_stats_suffix, ...
            sprintf('session-%s', session_tag));
        if ~exist(output_dir,'dir'), mkdir(output_dir); end

        spm_input_dir = fullfile(output_dir, 'spm_inputs');
        if ~exist(spm_input_dir,'dir'), mkdir(spm_input_dir); end

        %% --------- Find maps (.nii.gz) ----------
        pattern = fullfile(searchlight_dir, ...
            sprintf('sub-*_%s_mode-%s_cue-%s_classes-%d_radius-%s.nii.gz', ...
                    ['session-' session_tag], mode_tag, cue, n_cls, radius_str));

        map_files_gz = dir(pattern);
        fprintf('Found %d searchlight maps.\n', numel(map_files_gz));
        if isempty(map_files_gz), continue; end

        %% --------- Unzip and preprocess ----------
        chance_val = chance_fun(n_cls);
        smoothed_files = {};

        for i = 1:numel(map_files_gz)
            gz_path = fullfile(map_files_gz(i).folder, map_files_gz(i).name);

            [~,name,ext] = fileparts(gz_path);  % ext = '.gz'
            nii_path = fullfile(map_files_gz(i).folder, name);
            if ~exist(nii_path,'file')
                gunzip(gz_path, map_files_gz(i).folder);
            end

            % Load and center at chance
            V = spm_vol(nii_path);
            Y = spm_read_vols(V);
            Yc = Y - chance_val;

            % Save centered
            centered_path = fullfile(spm_input_dir, ['c_' name]);
            Vc = V; Vc.fname = centered_path;
            spm_write_vol(Vc, Yc);

            % Smooth
            smooth_path = fullfile(spm_input_dir, ['s_' name]);
            spm_smooth(centered_path, smooth_path, [fwhm fwhm fwhm]);

            smoothed_files{end+1,1} = [smooth_path ',1'];
        end

        %% --------- SPM one-sample t-test ----------
        matlabbatch = {};

        % Factorial design (one-sample)
        matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir};
        matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = smoothed_files;
        matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
        matlabbatch{1}.spm.stats.factorial_design.masking.im = 0;        
        matlabbatch{1}.spm.stats.factorial_design.masking.em = {mask_file};

        % Estimate
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(output_dir, 'SPM.mat')};
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

        % Contrast
        matlabbatch{3}.spm.stats.con.spmmat = {fullfile(output_dir, 'SPM.mat')};
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Decoding > chance';
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;

        spm_jobman('run', matlabbatch);
        fprintf('  -> Done. T-map: %s\n', fullfile(output_dir,'spmT_0001.nii'));
    end
end

fprintf('\nAll combinations finished.\n');
