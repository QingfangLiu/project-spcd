
% Unzips s2w2*.nii.gz files if not already unzipped
% Useful for prepping 2mm smoothed images before GLM estimation

% Unzips all s2w2*.nii.gz files in odor_scan directory (if needed)

scandir = fullfile(HomeDir, 'nii', SubID, 'func_PAID', 'odor_scan');

% Look for all gzipped 2mm smoothed odor scan files
gz_files = dir(fullfile(scandir, 's2w2Scan1_odor_scan_run_*.nii.gz'));

if isempty(gz_files)
    warning('No matching .nii.gz files found in %s', scandir);
end

for i = 1:length(gz_files)
    gz_path = fullfile(scandir, gz_files(i).name);

    % Remove .gz extension to get expected .nii name
    [~, name_noext] = fileparts(gz_path);  % strips .gz
    nii_path = fullfile(scandir, name_noext);

    if ~exist(nii_path, 'file')
        fprintf('Unzipping: %s\n', gz_path);
        gunzip(gz_path, scandir);
    else
        fprintf('Already unzipped: %s\n', nii_path);
    end
end
