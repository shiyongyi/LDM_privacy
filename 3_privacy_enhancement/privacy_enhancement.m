%% privacy enhancement

clc
clear
close all

format compact

%% dictionary threshold
Root = fullfile('.', 'generate_dictionary','dictionary'); % path of the dictionary
ori = readNPY(fullfile(Root, 'ori_dictionary.npy'));
syn = readNPY(fullfile(Root, 'syn_dictionary.npy'));

[N, ~] = size(syn); %number of synthetic images
[M, ~] = size(ori); %number of orginal images
thres = 0.000249; % threshold

% find the synthetic data with the L2 distance smaller  than the threshold
flag = ones(1,N);
for i = 1:N
    for j = 1:M
        L2 = rmse(syn(i,:),ori(j,:));
        if L2 < thres
            flag(i) = 0;
        end
    end
end

%% generate privacy enhanced samples
SavePath = fullfile('..', 'data', 'en_dataset'); %path of privacy enhanced dataset
fid = fopen(fullfile('.','generate_dictionary','dictionary','syn.txt'), 'rt');

if exist(SavePath,'dir') == 0
    mkdir(SavePath);
end

Name = cell(N,1);
for i = 1:N
    file = fgetl(fid);
    Name{i} = file;
end
fclose(fid);

for i = 1:N
    if flag(i) == 1
        data_enhanced = readNPY(Name{i}(4:end));
        Enhance = strsplit(Name{i},'\');
        Enhance = Enhance{end};
        writeNPY(data_enhanced,fullfile(SavePath, Enhance));
    end
end

