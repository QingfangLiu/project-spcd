

% this script goes back to copy the raw nifti files
% those in "func" folders have been preprocessed

%%

% organize images to different tasks
% need the following: T1 MPRAGE, 'place', 'character', 'odor_scan'
% 'odorpred', 'premeal', 'postmeal'
% note: some fMRI scans also have localizer sequence but not using them

for k = 1:length(pattern_list)
    
    pattern = pattern_list{k};
    nii_folder = fullfile(SubDir,'nii');
    tmp_out_folder = fullfile(nii_folder, 'func_raw', pattern);
    if ~exist(tmp_out_folder, 'dir')
        mkdir(tmp_out_folder);
    else
        continue; 
    end

    nifti_files = dir(fullfile(nii_folder, '*', ['*' pattern '*.nii']));
    nifti_files = nifti_files([nifti_files.bytes] > size_threshold);  % Filter the files based on the size limit (greater than 2 MB)

    % Copy all matching files to the individual folder
    parfor i = 1:length(nifti_files)
        
        nifti_file_name = nifti_files(i).name;
        source_file = fullfile(nifti_files(i).folder, nifti_file_name);
        destination_file = fullfile(tmp_out_folder, nifti_file_name);
        copyfile(source_file, destination_file);

        % also copy json files with the same names as the nifti files
        json_file_name = strrep(nifti_file_name,'nii','json');
        source_file_json = fullfile(nifti_files(i).folder, json_file_name);
        destination_file_json = fullfile(tmp_out_folder, json_file_name);
        copyfile(source_file_json, destination_file_json);

        % also compress all nifti files to save space
        % original files deleted automatically
        system(['gzip "' destination_file '"']);

    end

end

% in the end, zip the 1 and 2 folder to save some space
% deciding not to delete, because those *_SBRef files are not copied over
% and they may be useful for some special purposes

for f = 1:2
    folder_to_zip = fullfile(nii_folder, num2str(f));
    zip_name = [folder_to_zip '.zip'];       % Output zip file (same name with .zip extension)
    zip(zip_name, folder_to_zip);
    rmdir(folder_to_zip, 's');  % 's' = recursive deletion
end


% also zip all nifti files under 'func' folder to save space
% not very useful after combining across echoes
% or I can still work on nii.gz files without problem
% saved about 10GB per subj

func_nifti_files = dir(fullfile(nii_folder, 'func', '*', '*.nii'));
parfor i = 1:length(func_nifti_files)
    tmp_file_name = func_nifti_files(i).name;
    tmp_file = fullfile(func_nifti_files(i).folder, tmp_file_name);
    system(['gzip "' tmp_file '"']);
end