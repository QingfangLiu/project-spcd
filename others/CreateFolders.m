
HomeDir = '/Users/liuq13/SPCD_data';

% Number of main folders (Sub1 to Sub35)
numSubFolders = 40;

% Subfolder names to create within each Sub folder
subfolderNames = {'labchart', 'Scan1', 'Scan2'};

% Loop to create Sub folders and their respective subfolders
for i = 1:numSubFolders
    % Create the main Sub folder name
    mainFolderName = fullfile(HomeDir, sprintf('Sub%d', i));
    
    % Create the main Sub folder
    if ~exist(mainFolderName, 'dir')
        mkdir(mainFolderName);
    end
    
    % Create the subfolders within each Sub folder
    for j = 1:length(subfolderNames)
        subFolderPath = fullfile(mainFolderName, subfolderNames{j});
        if ~exist(subFolderPath, 'dir')
            mkdir(subFolderPath);
        end
    end
end

disp('Folders created successfully.');
