clc
clear
close all

%%LOAD
folder = "E:\Lab\Multi-channel EEG recording during motor imagery of different joints from the same limb";
dire = "\code\MI_2_processed_data\"; % Specify your folder path
folder = folder +dire;
filePattern = fullfile(folder, '*.mat'); % Look for all .mat files
matFiles = dir(filePattern); % Get the list of .mat files

for k = 1:1 %length(matFiles)
    baseFileName = matFiles(k).name;
    fullFileName = fullfile(folder, baseFileName);
    fprintf(1, 'Now reading %s\n', matFiles(k).name);
    data = load(fullFileName); % Load the .mat file
    % Optionally, you can store the data in a cell array or struct for later use:
    Volunteers{k} = data; % each 1xn matrix with one volunteer per colum, each volunteer has per line 'task_data', 'task_label', 'rest_data' 
end

% task_data: It contains all the task data of a subject with a size of (session number, trial number, channel number, sample points), e.g. (15, 40, 62, 800).
% task_label: It contains all the task label (“1” and “2” for MI of hand and elbow, respectively) of a subject with a size of (session number, trial number), e.g. (15, 40).
% rest_data: It contains a total of 300 trials from 4 rest sessions (75*4) with a size of (trial number, channel number, sample points), e.g. (300, 62, 800).


%%
v1 = Volunteers(1,1);
