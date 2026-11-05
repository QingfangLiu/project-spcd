function aggregate_group(run_dir, spm_dir, python_executable)
% Persistent SPM group models for the three updated value searchlights.
% Adapted from rsa_imagewise/aggregate_rsa_group.m: same GM masking,
% 6 mm smoothing, one-sample model, and positive t contrast. Only cohort,
% input/output routing, and failure handling differ. No collapsed phase fits.
%
% aggregate_group('/path/to/new_run', '/path/to/spm12', '/path/to/python')

if nargin < 3
    python_executable = 'python3';
end
wrapper = fullfile(fileparts(mfilename('fullpath')), 'run_searchlights.py');
command = strjoin({shell_quote(python_executable), '-B', shell_quote(wrapper), ...
    'check', '--run_dir', shell_quote(run_dir)}, ' ');
[code, output] = system(command);
fprintf('%s\n', output);
if code ~= 0
    error('Input validation failed; inspect check_report.csv before group inference.');
end
manifest_path = fullfile(run_dir, 'group_inputs.json');
inputs = jsondecode(fileread(manifest_path));
run_dir = fileparts(inputs.gm_mask); % Use absolute paths even if SPM changes directory.
manifest_path = fullfile(run_dir, 'group_inputs.json');
if numel(inputs.subjects) < 2
    error('At least two participants are required.');
end

group_root = fullfile(run_dir, 'group_spm');
if exist(group_root, 'dir')
    entries = dir(group_root);
    if any(~ismember({entries.name}, {'.', '..'}))
        error('Group output is not empty. Preserve/move that folder before a new group attempt: %s', group_root);
    end
end
addpath(spm_dir);
assert(exist('spm', 'file') == 2, 'SPM not found at the supplied path.');
spm('Defaults', 'fMRI');
spm_jobman('initcfg');
Vm = spm_vol(inputs.gm_mask);
M = spm_read_vols(Vm) > 0;
subsets = {'all', 'conditioned_only', 'preconditioned_only'};

for si = 1:numel(subsets)
    subset = subsets{si};
    paths = inputs.maps.(subset);
    assert(numel(paths) == numel(inputs.subjects), 'Participant count mismatch.');
    model_dir = fullfile(group_root, ['subset-' subset]);
    input_dir = fullfile(model_dir, 'spm_inputs');
    mkdir(input_dir);
    copyfile(manifest_path, fullfile(model_dir, 'group_inputs.json'));
    copyfile(fullfile(run_dir, 'analysis_config.json'), fullfile(model_dir, 'analysis_config.json'));
    fid = fopen(fullfile(model_dir, 'participants.txt'), 'w');
    assert(fid ~= -1, 'Could not write participant list.');
    fprintf(fid, '%s\n', inputs.subjects{:});
    fclose(fid);
    smoothed_files = cell(numel(paths), 1);

    for i = 1:numel(paths)
        expected = [inputs.subjects{i} '_model-odor_value_subset-' subset '.nii.gz'];
        [~, stem, extension] = fileparts(paths{i});
        assert(strcmp([stem extension], expected), 'Map/participant order mismatch.');
        unzipped = gunzip(paths{i}, input_dir);
        V = spm_vol(unzipped{1});
        assert(isequal(V.dim, Vm.dim) && max(abs(V.mat(:) - Vm.mat(:))) < 1e-5, ...
            'Input/GM grid mismatch: %s', paths{i});
        Y = spm_read_vols(V);
        assert(all(isfinite(Y(M))), 'Nonfinite coefficients: %s', paths{i});
        Y(~M) = NaN;
        [~, name, ~] = fileparts(unzipped{1});
        masked = fullfile(input_dir, ['c_' name '.nii']);
        smooth = fullfile(input_dir, ['s_' name '.nii']);
        V.fname = masked;
        spm_write_vol(V, Y);
        spm_smooth(masked, smooth, repmat(inputs.smoothing_fwhm_mm, 1, 3));
        smoothed_files{i} = [smooth ',1'];
    end

    spm_mat = fullfile(model_dir, 'SPM.mat');
    matlabbatch = {};
    matlabbatch{1}.spm.stats.factorial_design.dir = {model_dir};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = smoothed_files;
    matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 0;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {inputs.gm_mask};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
    matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat};
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
    matlabbatch{3}.spm.stats.con.spmmat = {spm_mat};
    matlabbatch{3}.spm.stats.con.delete = 0;
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = ...
        sprintf('RSA both | subset-%s | model-odor_value > 0', subset);
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    spm_jobman('run', matlabbatch);
    assert(exist(spm_mat, 'file') == 2 && ...
        exist(fullfile(model_dir, 'spmT_0001.nii'), 'file') == 2, 'SPM outputs missing.');
    fitted = load(spm_mat, 'SPM');
    assert(size(fitted.SPM.xX.X, 1) == numel(inputs.subjects) && ...
        abs(fitted.SPM.xX.erdf - (numel(inputs.subjects) - 1)) < 1e-6, ...
        'Saved SPM cohort/degrees of freedom do not match the prepared cohort.');
    fprintf('Saved n=%d group model: %s\n', numel(inputs.subjects), spm_mat);
end
end

function q = shell_quote(value)
% POSIX single-quote escaping for paths on macOS/Linux.
q = [char(39), strrep(char(value), char(39), ...
    [char(39), char(34), char(39), char(34), char(39)]), char(39)];
end
