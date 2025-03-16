
%%    
    % if the mean has been created
    mean_file = fullfile(SubDir,'nii', 'func_PAID', 'mean_fvol.nii');
    
    n = dir(fullfile(SubDir,'nii', 'func_PAID', '*', '*.nii')); 
    nruns = length(n);
    sum_mri_data = []; 
    total_volumes = 0;
    partial_sums = cell(nruns, 1);
    partial_volumes = zeros(nruns, 1);

parfor r = 1:nruns
    
    disp(['Processing run: ', n(r).name]);
    func_4d = fullfile(n(r).folder,n(r).name);
    
     V = spm_vol(func_4d);  % Use SPM to read the NIfTI file
     n_vols = numel(V);
     local_sum_mri_data = [];
     
    for i = 1:n_vols
        mri_data = spm_read_vols(V(i));
        
        % Initialize local_sum_mri_data in the first iteration
        if isempty(local_sum_mri_data)
            [dim1, dim2, dim3] = size(mri_data);
            local_sum_mri_data = zeros(dim1, dim2, dim3);
        end
        
        % Accumulate the sum for this run
        local_sum_mri_data = local_sum_mri_data + mri_data;
        
    end
    % Store the partial sum and volume count in cell arrays
    partial_sums{r} = local_sum_mri_data;
    partial_volumes(r) = n_vols;
    
end

% Combine partial sums from all runs
for r = 1:nruns
    if isempty(sum_mri_data)
        sum_mri_data = partial_sums{r};  % Initialize sum_mri_data on first use
    else
        sum_mri_data = sum_mri_data + partial_sums{r};
    end
    total_volumes = total_volumes + partial_volumes(r);
end

    mean_mri_data = sum_mri_data / total_volumes;
    func_4d = fullfile(n(1).folder,n(1).name);
    V = spm_vol(func_4d); 
    mean_volume_file = V(1);  % Use the header information from the last session    
    mean_volume_file.fname = mean_file;
    spm_write_vol(mean_volume_file, mean_mri_data);
    
