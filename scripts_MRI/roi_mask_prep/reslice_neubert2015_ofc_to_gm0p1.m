% Reslice selected Neubert 2015 OFC ROIs to the repo gm_0.1 mask grid.
%
% Click Run in the MATLAB editor after SPM is on the MATLAB path.
% Uses SPM Coregister: Reslice only. No transform is estimated.
% The output masks retain their original nonzero values; downstream ROI RSA
% scripts use mask > 0 when selecting voxels.

addpath('/Users/liuq13/Applications/spm12/');

this_file = mfilename('fullpath');
if isempty(this_file)
    project_root = '/Users/liuq13/project-spcd';
else
    script_dir = fileparts(this_file);
    project_root = fileparts(fileparts(script_dir));
end

% If needed, uncomment and edit this line:
% project_root = '/Users/liuq13/project-spcd';

if isempty(which('spm'))
    error(['SPM is not on the MATLAB path. Add SPM first, then rerun this ', ...
        'script.']);
end

roi_names = {
    'Left_11m.nii'
    'Right_11m.nii'
    'Left_14m.nii'
    'Right_14m.nii'
    'Left_Area_11.nii'
    'Right_Area_11.nii'
    'Left_Area_13.nii'
    'Right_Area_13.nii'
    'Left_47m.nii'
    'Right_47m.nii'
    'Left_47o.nii'
    'Right_47o.nii'
};

source_dir = fullfile(project_root, 'single_images');
ref_file = fullfile(project_root, 'masks', 'gm_0.1.nii');
output_dir = fullfile(project_root, 'masks', 'neubert2015_ofc_resliced_gm0p1');
staging_dir = fullfile(output_dir, 'source_for_spm');
output_prefix = 'gm0p1_';

assert(exist(source_dir, 'dir') == 7, 'Missing source directory: %s', source_dir);
assert(exist(ref_file, 'file') == 2, 'Missing reference image: %s', ref_file);

if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end
if exist(staging_dir, 'dir') ~= 7
    mkdir(staging_dir);
end

fprintf('Project root: %s\n', project_root);
fprintf('Reference: %s\n', ref_file);
fprintf('Source ROI directory: %s\n', source_dir);
fprintf('Output directory: %s\n', output_dir);
fprintf('Threshold convention for RSA: mask > 0\n');

copied_files = cell(numel(roi_names), 1);
for i = 1:numel(roi_names)
    src = fullfile(source_dir, roi_names{i});
    assert(exist(src, 'file') == 2, 'Missing ROI source file: %s', src);
    dst = fullfile(staging_dir, roi_names{i});
    copyfile(src, dst, 'f');
    copied_files{i} = dst;
end

source_with_vol = cell(size(copied_files));
for i = 1:numel(copied_files)
    source_with_vol{i} = [copied_files{i} ',1'];
end

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

matlabbatch = {};
matlabbatch{1}.spm.spatial.coreg.write.ref = {[ref_file ',1']};
matlabbatch{1}.spm.spatial.coreg.write.source = source_with_vol;
matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 0;
matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = 0;
matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = output_prefix;

fprintf('Running SPM Coregister: Reslice only for %d ROIs...\n', numel(roi_names));
spm_jobman('run', matlabbatch);

ref_vol = spm_vol(ref_file);
roi_col = cell(numel(roi_names), 1);
source_file_col = cell(numel(roi_names), 1);
resliced_file_col = cell(numel(roi_names), 1);
source_dim_col = cell(numel(roi_names), 1);
resliced_dim_col = cell(numel(roi_names), 1);
source_nonzero_col = zeros(numel(roi_names), 1);
resliced_nonzero_col = zeros(numel(roi_names), 1);
affine_max_abs_diff_col = zeros(numel(roi_names), 1);
dim_matches_ref_col = false(numel(roi_names), 1);
affine_matches_ref_col = false(numel(roi_names), 1);

fprintf('\nQC after reslicing:\n');
for i = 1:numel(roi_names)
    staged_output = fullfile(staging_dir, [output_prefix roi_names{i}]);
    final_output = fullfile(output_dir, [output_prefix roi_names{i}]);
    assert(exist(staged_output, 'file') == 2, 'SPM did not create: %s', staged_output);
    movefile(staged_output, final_output, 'f');

    src_vol = spm_vol(fullfile(source_dir, roi_names{i}));
    resliced_vol = spm_vol(final_output);
    src_data = spm_read_vols(src_vol);
    resliced_data = spm_read_vols(resliced_vol);

    affine_diff = max(abs(resliced_vol.mat(:) - ref_vol.mat(:)));
    dim_match = isequal(resliced_vol.dim, ref_vol.dim);
    affine_match = affine_diff < 1e-5;

    roi_col{i} = regexprep(roi_names{i}, '\.nii$', '');
    source_file_col{i} = fullfile(source_dir, roi_names{i});
    resliced_file_col{i} = final_output;
    source_dim_col{i} = sprintf('%dx%dx%d', src_vol.dim(1), src_vol.dim(2), src_vol.dim(3));
    resliced_dim_col{i} = sprintf('%dx%dx%d', resliced_vol.dim(1), resliced_vol.dim(2), resliced_vol.dim(3));
    source_nonzero_col(i) = nnz(src_data > 0);
    resliced_nonzero_col(i) = nnz(resliced_data > 0);
    affine_max_abs_diff_col(i) = affine_diff;
    dim_matches_ref_col(i) = dim_match;
    affine_matches_ref_col(i) = affine_match;

    fprintf('  %-15s source_nonzero=%5d  resliced_nonzero=%5d  dim_match=%d  affine_diff=%.3g\n', ...
        roi_col{i}, source_nonzero_col(i), resliced_nonzero_col(i), ...
        dim_match, affine_diff);
end

qc = table(roi_col, source_file_col, resliced_file_col, source_dim_col, ...
    resliced_dim_col, source_nonzero_col, resliced_nonzero_col, ...
    affine_max_abs_diff_col, dim_matches_ref_col, affine_matches_ref_col, ...
    'VariableNames', {'roi', 'source_file', 'resliced_file', 'source_dim', ...
    'resliced_dim', 'source_nonzero_voxels', 'resliced_nonzero_voxels', ...
    'affine_max_abs_diff_vs_ref', 'dim_matches_ref', 'affine_matches_ref'});

qc_file = fullfile(output_dir, 'neubert2015_ofc_reslice_qc.csv');
writetable(qc, qc_file);

fprintf('\nSaved resliced masks: %s\n', output_dir);
fprintf('Saved QC table: %s\n', qc_file);
fprintf('Use these masks with ROI RSA via --roi_mask masks/neubert2015_ofc_resliced_gm0p1/<mask>.nii\n');
