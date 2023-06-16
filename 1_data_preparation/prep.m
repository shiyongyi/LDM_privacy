% The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge organized by Mayo Clinic   
% (I am authorized to share this dataset but you can ask Mayo Clinical directly 
% for permission at the URL: https://www.aapm.org/GrandChallenge/LowDoseCT/, 
% which is for public data sharing).

clc
clear
close all

format compact

FullRoot = fullfile('..','MayoDataset'); % path of Mayo Clinic Dataset
SaveRoot = fullfile('..','data','ori_dataset'); % path of original dataset.

if exist(SaveRoot,'dir') == 0
    mkdir(SaveRoot);
end

Files = dir(FullRoot);
Files = Files(3:end);

L = length(Files);

for i = 1:L  
    SlicePath = fullfile(FullRoot,Files(i).name,'full_3mm');
    Slices = dir(SlicePath);
    Slices = Slices(3:end);
    S = length(Slices);
    for j = 1:S
        Image = dicomread(fullfile(SlicePath,Slices(j).name)); % read image
        Image = double(Image) - 1024; % translate to CT value
        Image = Image/1000*0.192+0.192; % CT value to attenuation
        Image(Image<0) = 0;  
        Image = single(Image);  % single type
        writeNPY(Image,fullfile(SaveRoot, [Files(i).name, '_', num2str(j-1), '.npy']));       
    end
end