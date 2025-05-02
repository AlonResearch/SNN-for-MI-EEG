
%% Visualization.m
% This code is about for the representation of spactial information and spectral analysis based on scalp topography and ERSP in selected session.
% (ERSP: Event-related spectral perturbation)

% Please set the download directory of your computer for visualization.
% Please add the bbci toolbox in the 'Reference_toolbox' folder
%%
%% Scalp Topography
% Initializing
clear; clc; close all;

% Directory
% `Download_folder'is download directory of user's computer, so you have to change the directory correctly.
dd='E:\Lab\Scripts\Sample_Data';


% The filelist can be downloaded from sampleData.
% It is one of the example data.
filelist = {'\SampleData_Acc_ArmReaching'};

% setting time: 0 ~ 4 seconds
ival=[0 4000];

% select channels: A total of EEG channels
selected_channels = [1:31, 36:64];

% Load .mat file to EEG pre-processing
[cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{1}]);

% band pass filtering, order of 2, range of [8-15] Hz (mu-band)
cnt=proc_filtButter(cnt, 2, [8 13]);

cnt=proc_selectChannels(cnt, selected_channels);

% Making a epoched data according to the time interval
epo=cntToEpo(cnt,mrk,ival);

classes=size(epo.className,2);
trial=size(epo.x,3)/2/(classes-1);

% Set the number of rest trial to the same as the number of other classes trial.
for ii =1:classes
    if strcmp(epo.className{ii},'Rest')
        epoRest=proc_selectClasses(epo,{epo.className{ii}});
        epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
        epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
    else
        epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
        % random sampling
        epo_check(ii).x=datasample(epo_check(ii).x,trial,3,'Replace',false);
        epo_check(ii).y=datasample(epo_check(ii).y,trial,2,'Replace',false);
    end
end

if classes<3
    epo_check(size(epo_check,2)+1)=epoRest;
end

% Concatenate the classes
for ii=1:size(epo_check,2)
    if ii==1
        concatEpo=epo_check(ii);
    else
        concatEpo=proc_appendEpochs(concatEpo,epo_check(ii));
    end
end

% Set ival_scalps for plotting the scalp map at each time point:
% 0~1000, 1000~2000, 2000~3000, 3000~4000 (ms)
ival_scalps= 0:1000:4000;

% Set the total time of each trial including the baseline time.
t_ival = [-500 4000];
epo = concatEpo;
epo = cntToEpo(cnt, mrk, t_ival);
fv = proc_rectifyChannels(epo);
fv = proc_movingAverage(fv, 200, 'centered');
fv = proc_baseline(fv, [-500 0]);
erp = proc_average(fv);
clab = epo.clab;

% Visualization
figure();
scalpEvolutionPlusChannel(erp, mnt, clab, ival_scalps);
colormap('jet')
hold on;

% If you want to plot all subject's scalp topography, use
% Revision_All_Topography.m file.

%%
%% ERSP
% 1. Install EEGLAB. (https://sccn.ucsd.edu/eeglab/index.php) and run 'eeglab' toolbox.
% 2. Load data: [Import data] - [Using EEGLAB functions and plugins] - [From Brain Vis Rec .vhdr file]
% 3. Extract channel: [Edit] - [Selected data] % Choose the channel that you want to remove. The channel should be EOG and EMG channels.
% 4. Set channel location: [Edit] - [Channel locations]
% 5. Preprocessing: [Tools] - [Filter the data] - [Basic FIR filter(legacy)] % Put 8 in 'Lower edge of the frequency pass band(Hz)' 13 in 'Higher edge of the frequency pass band(Hz)'  and FIR filter oder is 5.
% 6. Divide data in each class: [Tools] - [Extract epochs] % Select the class using the trigger number. The information on the trigger number is as follows.
% Trigger information:
% Arm reaching - 'S 11': Forward, 'S 21': Backward, 'S 31': Left, 'S 41': Right, 'S 51': Up, 'S 61': Down, 'S  8': Rest
% Hand grasping - 'S 11': Cylindrical, 'S 21': Spherical, 'S 61': Lumbrical, 'S  8': Rest
% Wrist twisting - 'S 91': Left,  'S101': Right, 'S  8': Rest
% 7. Put the epoch you want to analyze in 'Time-loching event type(s)'.
% 8. Put the time of each trial (0 ~ 4s) in 'Epoch limits [satrt, end] in seconds.
% 9. Plot ERSP: [Plot] - [Time-frequency] - [Channel time-frequency] / Set the channel number you want to analyze.
%бщ
eeglab
% if you have already installed eeglab, please drag the "eeglab" and run the code (F9)

%%
% If you need any help for visualization, please contact this e-mail: bh_kwon@korea.ac.kr


