function run_second_image_value_svc(repo_root, spm_dir)
% Peak-level SVC for the nine synced second-image value RSA group models.
% Example (after adding this directory to the MATLAB path):
%   run_second_image_value_svc('/Users/liuq13/project-spcd','/Applications/spm12')
% Uses native SPM12 and reproduces the existing n30 OFC SVC before writing.
% Does not relocate or modify the source SPM.mat or require subject maps.
if nargin < 1
    repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2, spm_dir = '/Applications/spm12'; end
addpath(spm_dir);
assert(strcmp(spm('Ver'), 'SPM12'), 'This workflow requires SPM12.');
out_base = fullfile(repo_root, 'rsa_roi', 'probe_imagewise_value');
reference_dir = fullfile(out_base, 'searchlight_svc_n30_ofc_v1');
reference_meta = jsondecode(fileread(fullfile(reference_dir, 'analysis_metadata.json')));
roi_names = {'Medial_OFC', 'Lateral_OFC'};

% Reproduce every numeric field in the established reference summary.
[baseline, ~] = calculate('spm_value_n30');
reference = readtable(fullfile(reference_dir, 'peak_svc_summary.csv'));
assert(isequal(baseline.ROI, reference.ROI));
assert(max(abs(table2array(baseline(:,2:end)) - ...
    table2array(reference(:,2:end))), [], 'all') < 1e-8, ...
    'The previous OFC SVC results were not reproduced.');
fprintf('Reproduced both previous n30 OFC SVC rows.\n');

models = {'cue_value', 'signed_difference', 'absolute_difference'};
subsets = {'', '_conditioned_only', '_preconditioned_only'};
combined = table();
for m = 1:numel(models)
    for s = 1:numel(subsets)
        suffix = [models{m} subsets{s}];
        source_name = ['spm_value_n30_second_image_' suffix];
        [summary, meta] = calculate(source_name);
        meta.value_model = models{m};
        subset = subsets{s};
        if isempty(subset), subset = 'all'; else, subset = subset(2:end); end
        meta.cue_subset = subset;
        meta.session = 'both';
        meta.image_scope = 'second';
        meta.spm_version = spm('Version');
        meta.analysis_script = 'scripts_MRI/rsa_imagewise/run_second_image_value_svc.m';
        meta.checks = {'Previous n30 OFC SVC summary reproduced with native SPM12.', ...
            'ROI and group image grids identical; no ROI resampling.', ...
            'Participant IDs match the established n30 analysis.'};
        target = fullfile(out_base, ['searchlight_svc_n30_ofc_second_image_' suffix '_v1']);
        if ~exist(target, 'dir'), mkdir(target); end
        writetable(summary, fullfile(target, 'peak_svc_summary.csv'));
        fid = fopen(fullfile(target, 'analysis_metadata.json'), 'w');
        assert(fid ~= -1);
        fprintf(fid, '%s\n', jsonencode(meta));
        fclose(fid);
        summary = addvars(summary, repmat(models(m),2,1), repmat({subset},2,1), ...
            'Before', 1, 'NewVariableNames', {'value_model','cue_subset'});
        combined = [combined; summary]; %#ok<AGROW>
        fprintf('%s: medial p=%.9f; lateral p=%.9f\n', suffix, ...
            summary.peak_p_FWE_SVC(1), summary.peak_p_FWE_SVC(2));
    end
end
target = fullfile(out_base, 'searchlight_svc_n30_ofc_second_image_v1');
if ~exist(target, 'dir'), mkdir(target); end
writetable(combined, fullfile(target, 'peak_svc_summary.csv'));

    function [summary, meta] = calculate(source_name)
        source = fullfile(repo_root, 'temp_maps', source_name);
        loaded = load(fullfile(source, 'SPM.mat'), 'SPM');
        SPM = loaded.SPM;
        assert(numel(SPM.xCon) == 1 && strcmp(SPM.xCon.STAT, 'T') && ...
            isequal(SPM.xCon.c, 1), 'Expected one positive T contrast.');
        assert(isequal(size(SPM.xX.X), [30 1]) && all(SPM.xX.X == 1));
        assert(abs(SPM.xX.erdf - 29) < 1e-8);
        subjects = regexp(cellstr(SPM.xY.P), 'sub-\d+', 'match', 'once');
        assert(isequal(subjects(:), reference_meta.participant_ids(:)));
        Vt = spm_vol(fullfile(source, 'spmT_0001.nii'));
        Vm = spm_vol(fullfile(source, 'mask.nii'));
        assert(isequal(Vt.dim, Vm.dim) && max(abs(Vt.mat-Vm.mat),[],'all') < 1e-8);
        tmap = spm_read_vols(Vt);
        group_mask = spm_read_vols(Vm);
        fwhm = SPM.xVol.FWHM;
        df = [SPM.xCon.eidf SPM.xX.erdf];
        rows = cell(2,13);
        meta = struct();
        meta.source_group_model = ['temp_maps/' source_name '/SPM.mat'];
        meta.contrast = SPM.xCon.name;
        meta.n_participants = size(SPM.xX.X,1);
        meta.participant_ids = subjects;
        meta.df_error = df(2);
        meta.smoothness_FWHM_voxels = fwhm;
        meta.roi_masks = reference_meta.roi_masks;
        meta.spatial_correction = reference_meta.spatial_correction;
        meta.regional_correction = reference_meta.regional_correction;
        meta.model_correction = 'None across the nine model/subset analyses; exploratory checks.';
        for r = 1:2
            name = roi_names{r};
            Vr = spm_vol(fullfile(repo_root, reference_meta.roi_masks.(name)));
            assert(isequal(Vr.dim,Vt.dim) && max(abs(Vr.mat-Vt.mat),[],'all') < 1e-8);
            roi = spm_read_vols(Vr);
            native = isfinite(roi) & roi ~= 0;
            overlap = native & isfinite(group_mask) & group_mask ~= 0 & isfinite(tmap);
            indices = find(overlap);
            assert(~isempty(indices));
            values = tmap(indices);
            [peak, ix] = max(values);
            [x,y,z] = ind2sub(Vt.dim, indices(ix));
            xyz = Vt.mat * [x;y;z;1];
            R = spm_resels(fwhm, Vr, 'I');
            S = numel(indices);
            corrected = spm_P(1,0,peak,df,'T',R,1,S);
            critical = spm_uc(0.05,df,'T',R,1,S);
            uncorrected = 1-spm_Tcdf(peak,df(2));
            cdt = spm_u(0.001,df,'T');
            rows(r,:) = {name, nnz(native), S, peak, xyz(1), xyz(2), xyz(3), ...
                corrected, uncorrected, critical, R(4), nnz(values>cdt), ...
                nnz(values>critical)};
            meta.roi_resels.(name) = R;
        end
        summary = cell2table(rows, 'VariableNames', ...
            {'ROI','native_roi_voxels','analysis_overlap_voxels','peak_T', ...
            'MNI_x','MNI_y','MNI_z','peak_p_FWE_SVC','peak_p_uncorrected', ...
            'critical_T_FWE_05','resel_volume','voxels_above_CDT_001','voxels_above_FWE_05'});
    end
end
