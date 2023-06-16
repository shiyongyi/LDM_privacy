function rmseMC = rmse3d(im0,im)
%
% RMSE of all results
%
% Input variables:
%           'im0'                                 reference image, 3D
%           'im'                                   image, 3D
% Output variables:
%           'rmseMC'                          rmse for each channel image
%           'rmsetotal'                        rmse for total image
%
% Yanbo Zhang
% University of Massachusetts Lowell
% yanbozhang007@gmail.com
% 2015-08-16

[nx, ny] = size(im0);


rmseMC = sqrt(sum(sum((im0 - im).^2))/nx/ny);    

