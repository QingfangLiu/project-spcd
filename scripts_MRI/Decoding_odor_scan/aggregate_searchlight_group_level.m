
% aggregate_odor2_searchlight_group_level_spm.m
%
% SPM12 group analysis for odor localizer (2-odor) searchlight maps.
%
% For each analysis option:
%   odor_scan_2odors/<opt>/sub-*_searchlight.nii.gz
%
% - Centers each map at chance (0.5), applies GM mask, smooths
% - Runs a one-sample t-test in SPM on (acc - chance)
% - Uses explicit GM mask: /SPCD_data/masks/gm_0.1.nii
% - Stores results in:
%   odor_scan_2odors/<opt>/group_stats_spm/

%% ---------- Paths & settings ----------
addpath('/Users/liuq13/Applications/spm12/');
base_dir   = '/Users/liuq13/SPCD_data/searchlight_maps';
task_folder  = 'odor_scan_2odors';
opt_list     = {'across_trials', 'across_voxels', 'none'}; 
opt_list     = {'across_trials', 'none'}; 

group_stats_suffix = 'group_stats_spm';

mask_file = '/Users/liuq13/SPCD_data/masks/gm_0.1.nii';
fwhm      = 6;
chance_val = 0.5;  % 2-class decoding

fprintf('Using mask: %s\n', mask_file);

%% ---------- Initialize SPM ----------
spm('Defaults','fMRI');
spm_jobman('initcfg');

%% ---------- Loop over analysis options ----------
for oi = 1:numel(opt_list)
    opt = opt_list{oi};
    fprintf('\n============================================\n');
    fprintf('>>> SPM group analysis | %s\n', opt);
    fprintf('============================================\n');

    % Directories for this option
    opt_dir    = fullfile(base_dir, task_folder, opt);
    input_dir  = opt_dir;
    output_dir = fullfile(opt_dir, group_stats_suffix);

    if ~exist(output_dir, 'dir'), mkdir(output_dir); end

    spm_input_dir = fullfile(output_dir, 'spm_inputs');

    if ~exist(spm_input_dir, 'dir'), mkdir(spm_input_dir); end

    fprintf('  Input dir : %s\n', input_dir);
    fprintf('  Output dir: %s\n', output_dir);

    %% ---------- Find maps (.nii.gz) ----------
    pattern = fullfile(input_dir, 'sub-*_*searchlight.nii.gz');   % sub-01_searchlight.nii.gz
    map_files_gz = dir(pattern);
    fprintf('  Found %d searchlight maps.\n', numel(map_files_gz));
    if isempty(map_files_gz)
        fprintf('  No maps found for %s, skipping.\n', opt);
        continue;
    end

    %% ---------- Load mask once ----------
    VM = spm_vol(mask_file);
    M  = spm_read_vols(VM) > 0;

    %% ---------- Unzip, center at chance, smooth ----------
    smoothed_files = {};

    for i = 1:numel(map_files_gz)
        gz_path = fullfile(map_files_gz(i).folder, map_files_gz(i).name);

        [~, name, ext] = fileparts(gz_path);  % ext = '.gz'
        nii_path = fullfile(map_files_gz(i).folder, name);

        % Unzip if needed
        if ~exist(nii_path, 'file')
            fprintf('  Unzipping %s\n', gz_path);
            gunzip(gz_path, map_files_gz(i).folder);
        end

        % Load and center at chance
        V = spm_vol(nii_path);
        Y = spm_read_vols(V);

        if ~isequal(size(Y), size(M))
            error('Size mismatch between data and mask for %s', nii_path);
        end

        % Mask: set outside-GM voxels to NaN
        Y(~M) = NaN;

        % Center at chance
        Yc = Y - chance_val;

        % Save centered image
        centered_path = fullfile(spm_input_dir, ['c_' name]);
        Vc       = V;
        Vc.fname = centered_path;
        spm_write_vol(Vc, Yc);

        % Smooth centered image
        smooth_path = fullfile(spm_input_dir, ['s_' name]);
        spm_smooth(centered_path, smooth_path, [fwhm fwhm fwhm]);

        smoothed_files{end+1,1} = [smooth_path ',1'];
    end

    %% ---------- SPM one-sample t-test ----------
    matlabbatch = {};

    % Factorial design (one-sample t-test)
    matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = smoothed_files;
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 0;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {mask_file};
    
    % Estimate
    matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(output_dir, 'SPM.mat')};
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    % Contrast: Decoding > chance
    matlabbatch{3}.spm.stats.con.spmmat = {fullfile(output_dir, 'SPM.mat')};
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Decoding > chance';
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
   
    %% ---------- Run batch ----------
    spm_jobman('run', matlabbatch);
    fprintf('  -> Done for %s. T-map in: %s\n', opt, fullfile(output_dir, 'spmT_0001.nii'));
    
end

fprintf('\nAll odor_scan_2odors options finished.\n');

