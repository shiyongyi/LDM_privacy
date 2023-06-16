%% privacy evaluation

clc
clear
close all

format compact

%% L2 distance calculation
oriRoot = fullfile('..', 'data', 'ori_dataset'); % path of original dataset
synRoot = fullfile('..', 'data', 'syn_dataset'); % path of synthetic dataset or privacy enhanced dataset

oriFile = dir(oriRoot);
oriFile = oriFile(3:end);

synFile = dir(synRoot);
synFile = synFile(3:end);

N = length(oriFile);
M = length(synFile);
volumeX = 512;
volumeY = 512;
oriImage = zeros(volumeX,volumeY,N);
matImage = zeros(volumeX,volumeY,M*2);
Distance = zeros(1,M);
for i = 1:N
    oriImage(:,:,i) = readNPY(fullfile(oriRoot, oriFile(2).name));
end

for i = 1:M
    i
    syn = readNPY(fullfile(synRoot, synFile(i).name));
        
    L2 = zeros(1,N);
    for j = 1:N
        L2(j) = rmse(syn, oriImage(:,:,j));
    end
	L2(L2==0) = 1;
    Distance(i) = min(L2);
    index = find(L2 == min(L2));
    matImage(:,:,i) = syn;
    matImage(:,:,M+i) = oriImage(:,:,index(1));
end

figure, histogram(Distance,100);
