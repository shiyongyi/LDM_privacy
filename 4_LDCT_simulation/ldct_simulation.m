%% #######################Notes on Script############################
% General comments:
% 	
% This code is used for fan-beam CT iterative reconstruction.
% The order subset method is used to accelerate the convergence.
% ray-driven is used in forward proejction and pixel-driven is used in backprojection
% And the detector is circular in this script
clc
clear
close all

format compact

dataset = 'syn';% original dataset: 'ori'; synthetic_dataset: 'syn'.
root = fullfile('..', 'data', [dataset,'_dataset']);
ldct = fullfile('..', 'data', [dataset,'_ldct']);
sino = fullfile('..', 'data', [dataset,'_sino']);
test_ldct = fullfile('..', 'data', 'test_ldct');
test_sino = fullfile('..', 'data', 'test_sino');

if exist(ldct,'dir') == 0
    mkdir(ldct);
end
if exist(sino,'dir') == 0
    mkdir(sino);
end
if exist(test_ldct,'dir') == 0
    mkdir(test_ldct);
end
if exist(test_sino,'dir') == 0
    mkdir(test_sino);
end

Files = dir(root);
Files = Files(3:end);
Len = length(Files);

%% input parameters
photon = 1e4;
iteration = 20;
subset = 20;

%geometry
NumberofViews = 1152;
NumberofDetector = 672;

VolumeX = 512;
VolumeY = 512;

Fov = VolumeX*0.8;
IsoSource = 570;
MaxAngle = 0.4526;
MinAngle = -0.4560;

% Source location
Angle = [1:NumberofViews]*2*pi/NumberofViews;

% the parameters about forwardprojection and backprojection
ParaForward = [NumberofDetector, Fov, IsoSource, MaxAngle, MinAngle];
ParaBack = [VolumeX, VolumeY, Fov, IsoSource, MaxAngle, MinAngle];

for i = 1:Len
    Image = readNPY(fullfile(root,Files(i).name));   
    Projection = ForwardProjection(double(Image), Angle, ParaForward);
    Projection = Projection.data/10;
    sinogram = Projection/10; %normalize the sinogram to 0-1
    
    ProjAnalysis = photon * exp(-Projection);
    NoiseProjection = poissrnd(ProjAnalysis);
    B =(NoiseProjection==0);
    NoiseProjection = NoiseProjection+B;
    RealProjection = -log(NoiseProjection/photon);
    sino_noise = RealProjection/10;
    Image_noise = SART(RealProjection,Angle,ParaForward,ParaBack,iteration,subset);
    
    Image = single(Image);
    sinogram = single(sinogram);
    Image_noise = single(Image_noise)*10;
    RealProjection = single(RealProjection)/10;

    if strcmp(Files(i).name(1:4),'L506')    
        writeNPY(Image,fullfile(test_ldct, [Files(i).name(1:end-4),'_target.npy']));
        writeNPY(sinogram,fullfile(test_sino, [Files(i).name(1:end-4),'_target.npy']));
        writeNPY(Image_noise, fullfile(test_ldct, [Files(i).name(1:end-4),'_input.npy']));
        writeNPY(RealProjection, fullfile(test_sino, [Files(i).name(1:end-4),'_input.npy']));
    else
        writeNPY(Image,fullfile(ldct, [Files(i).name(1:end-4),'_target.npy']));
        writeNPY(sinogram,fullfile(sino, [Files(i).name(1:end-4),'_target.npy']));
        writeNPY(Image_noise, fullfile(ldct, [Files(i).name(1:end-4),'_input.npy']));
        writeNPY(RealProjection, fullfile(sino, [Files(i).name(1:end-4),'_input.npy']));
    end
end



