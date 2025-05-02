% EEG BIDS Feature extraction pipeline 
% and exported proceesed data + features in .mat fileby Arthur Alonso 2024:
%  MI data organized into (15, 40, 62, 800), (15, 40, 1), and (300, 62, 800)
%             (Session, Trial, Channel, Time) (Session, Trial, Label) (Rest, Channel, Time)


% %% This part is used to organize the data and save it as a .mat file for subsequent classifier construction
% % Data loading, filtering, extraction, saving (subject_01.mat) 

clear
clc
loadDir = 'E:\Lab\MI_3class_1limb\MI\';
saveDir = 'E:\Lab\MI_3class_1limb\MI\Derivatives\';


%
% LIST ALL THE SUBJECTS
%
ListSub = dir(fullfile(loadDir, 'sub-*'));  % List all the volunteers on the sourcedatafolder with format sub-*
TotSub = size(ListSub, 1); % Total number of subjects

NumSub = 1;
% for NumSub = 1: TotSub % Iterate on the Subject list
    NameSub = ListSub(NumSub).name;
    PathSes = [loadDir, NameSub, '\']; % Path to the sessions folder
    if ~exist(PathSes) % if dosent exist create a folder
        mkdir(PathSes)
    end



    % Tech folder
    NameTech = 'eeg';


    % Save the file
    PathSaveFile = [saveDir,NameSub,'\', NameTech, '\'];
    if ~exist(PathSaveFile) % if dosent exist create a folder
        mkdir(PathSaveFile)
    end
    NameFile = [NameSub, '.mat'];
    checkPath = ["PathSaveFile", NameFile]; % the path of the file being created
    disp(checkPath) % display the path of the file being created
    %if exist("PathSaveFile", 'file') % if the file already exists, skip the processing
        % continue; when used with a loop;
        a = 0; % do nothing
    %else

        %
        % Now process the data
        %

        % Create matrixes to store the data
        task_data = zeros(15, 40, 62, 800);
        task_label = zeros(15, 40, 1);  
        rest_data = zeros(300, 62, 800);
        rest_label = 3*ones(300, 1);
            

        ListSes=dir(fullfile(PathSes));  % List all the sessions for a given volunteer
        TotSes=size(ListSes,1); % the first two are "." and ".." 
        %NumSes = 3;
        for NumSes = 3: TotSes % Iterate on the Sessions list, the first two are "." and ".." so begin with index 3
            NameSes = ListSes(NumSes).name;
            PathTech = [PathSes, NameSes, '\']; % Path to the techniques folder
            PathFile = [PathTech, NameTech, '\'];


        % part 1 Load data
        % Load mat data - get hand and elbow data
        % Each 8s (3s before the prompt and 5s after the prompt) is a segment, and they cannot overlap, including rest
        % Therefore, the task state is a total of 300 samples each, and the resting state is 160 samples each.
        
            
            % Get MI data
            ListMIFiles = dir([PathFile, '*motorimagery_eeg.set']); % List the .set files with the data
            if ~isempty(ListMIFiles)
                EEG = pop_loadset('filename',ListMIFiles(1).name,'filepath',ListMIFiles(1).folder);
                EEG = pop_resample( EEG, 200); % Directly downsample to 200Hz
                data=double(EEG.data);
                % filter
                EEG = pop_eegfiltnew(EEG,  7, []);
                EEG = pop_eegfiltnew(EEG, [], 40);
                event=EEG.event;
                for j = 1:40
                    latency = event(1,j).latency;
                    task_data(NumSes-2, j, :, :) = data(:,latency:latency+799);
                    if strcmp(class(event(1,j).type), 'char')
                        task_label(NumSes-2, j, 1) = str2num(event(1,j).type);
                    else
                        task_label(NumSes-2, j, 1) = event(1,j).type;
                    end
                end

            end

            % Get rest data
            ListRestFiles = dir([PathFile, '*rest_eeg.set']); % List the .set files with the data
            if ~isempty(ListRestFiles)                
                EEG = pop_loadset('filename',ListRestFiles(1).name,'filepath',ListRestFiles(1).folder);
                EEG = pop_resample( EEG, 200); % Directly downsample to 200Hz
                data=double(EEG.data);
                % filter
                EEG = pop_eegfiltnew(EEG,  4, []);
                EEG = pop_eegfiltnew(EEG, [], 45);

                latency = 1;
                for j = 1:75
                    rest_data(j+(NumSes-3)*75, :, :) = data(:,latency:latency+799);
                    latency = latency + 849;
                end
            end
        end
    save([saveDir,NameSub,'\', NameFile], 'task_data', 'task_label', 'rest_data')

    %Features extraction


    % Calculate the percentage of each of those frequencies from the power spectrum of the EEG signal for each epoch
    % Hz: Delta < 3, Theta 4–7, Alpha 8–14, Mu 8–12, Beta 15–30, Gamma > 30.
    % Initialize frequency bands
        gamma_band = [30, 45];
        beta_band = [15, 30];
        mu_band = [8, 12];
        alpha_band = [8, 14];
        theta_band = [4, 7];
        delta_band = [0.5, 3];

    % Initialize arrays to store the power percentages
    gamma_power = zeros(15, 40);
    beta_power = zeros(15, 40);
    mu_power = zeros(15, 40);
    alpha_power = zeros(15, 40);
    theta_power = zeros(15, 40);
    delta_power = zeros(15, 40);

    for session = 1:15 % To be included Include the label of the session
        for trial = 1:40
            % Get the data for the current trial
            eeg_data = squeeze(task_data(session, trial, :, :));
            

             % Calculate the power spectrum
            [pxx, f] = pwelch(eeg_data', [], [], [], 200);
           

            total_power = sum(pxx, 1); % Calculate the total power

            gamma_power(session, trial) = sum(pxx(f >= gamma_band(1) & f <= gamma_band(2), :), 1) ./ total_power;
            beta_power(session, trial) = sum(pxx(f >= beta_band(1) & f <= beta_band(2), :), 1) ./ total_power;
            mu_power(session, trial) = sum(pxx(f >= mu_band(1) & f <= mu_band(2), :), 1) ./ total_power;
            alpha_power(session, trial) = sum(pxx(f >= alpha_band(1) & f <= alpha_band(2), :), 1) ./ total_power;
            theta_power(session, trial) = sum(pxx(f >= theta_band(1) & f <= theta_band(2), :), 1) ./ total_power;
            delta_power(session, trial) = sum(pxx(f >= delta_band(1) & f <= delta_band(2), :), 1) ./ total_power;
            % Calculate the power in each frequency band
        
            %Calculate the mean, standard deviation, and variance of the eeg signal for each epoch

            mean_eeg = mean(eeg_data, 2);
            std_eeg = std(eeg_data, 0, 2);
            var_eeg = var(eeg_data, 0, 2);

            % Verify more features on papers**



        end
    

    % Create a features matrix
    features_matrix = cat(3, delta_power, theta_power, alpha_power, mu_power, beta_power, gamma_power, mean_eeg, std_eeg, var_eeg);     
    % Save the features matrix
    save([saveDir, NameSub, '\', NameFile], 'features_matrix', '-append');
    %end
%end







  