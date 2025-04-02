
% this script is to copy 1.zip and 2.zip files under each ppt of SPCD
% the two zip files are prepared after running the
% "Do_1_half_recopy_raw_nifti" script
% this is copied to external drive
% 
% The two zipped files contain nifti files for all runs and related SBref files, json
% files
% But I don't expect that I still need those files
% because BIDS version of this dataset has been uploaded to OpenNeuro

% another backup can be found on NIDA lab server with the original and
% uncleaned DCM files (uploaded by Taaseen)

% if I ever want to reuse those zipped files, go back to Do_1* and
% Do_1_half* scripts to continue.

% rerun this script will skip those already done

% Set your source and external drive base paths
sourceRoot = '/Users/liuq13/SPCD_data/';
destinationRoot = '/Volumes/QF10TB/SPCD_data/';

% List all entries in the source directory
entries = dir(sourceRoot);

% Loop through subject folders
for i = 1:length(entries)
    entry = entries(i);
    
    % Check if it's a folder and matches the pattern 'Sub*_done'
    if entry.isdir && startsWith(entry.name, 'Sub') && endsWith(entry.name, '_done')
        subjectFolder = entry.name;
        sourceNiiPath = fullfile(sourceRoot, subjectFolder, 'nii');
        destNiiPath = fullfile(destinationRoot, subjectFolder, 'nii');

        % Ensure destination folder exists
        if ~exist(destNiiPath, 'dir')
            mkdir(destNiiPath);
        end

        % Files to move
        zipFiles = {'1.zip', '2.zip'};

        for j = 1:length(zipFiles)
            srcFile = fullfile(sourceNiiPath, zipFiles{j});
            destFile = fullfile(destNiiPath, zipFiles{j});
            
            % Move if exists
            if isfile(srcFile)
                movefile(srcFile, destFile);
                fprintf('Moved: %s -> %s\n', srcFile, destFile);
            else
                fprintf('Not found: %s\n', srcFile);
            end
        end
    end
end
