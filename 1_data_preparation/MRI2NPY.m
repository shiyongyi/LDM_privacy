clc
clear
close all

format compact

root = fullfile('..','MRI_IXI'); % path of MRI IXI Dataset
saved = fullfile('..','data','ori_dataset'); % path of original dataset.

files = dir(root);
files = files(3:end);
len = length(files);
b = zeros(1,len);
flag = 0;
Max = 3000;
for i = 1:len
    path = fullfile(root,files(i).name,'T2','NIfTI');
    patient = dir(path);
    name = patient(3).name;
    %gunzip(fullfile(path,name))
    data = niftiread(fullfile(path,name));
    data = double(data);
    [~,~,Size] = size(data);
    if Size < 100
        continue;
    end
    for j = 1:50
        flag = flag + 1;
        M = floor(Size/2);
        B = M - 25;
        image = data(:,:,B+j);
        image(image>Max) = Max;
        image = image/Max;
        low = image(1:4:end,1:4:end);
        writeNPY(single(low),fullfile(saved,['s',num2str(flag),'_input.npy']));
        writeNPY(single(image),fullfile(saved,['s',num2str(flag),'_target.npy']));
    end
end