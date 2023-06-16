%% %%%%%%%%%%%%%%%this function is used to SART %%%%%%%%%%%%%%%%%%
function output = SART(RealProjection,Angle,ParaForward,ParaBack,iteration,subset)

%% %%%%%%%%%%%%%%%% calculate the denominator %%%%%%%%%%%%%%%%%%%%%%
[~,NumberofViews] = size(RealProjection);
ImageOnes = ones(ParaBack(1), ParaBack(2));
ProjectionOnes = ForwardProjection(ImageOnes, Angle, ParaForward);
FidelityDenominator = BackProjection(ProjectionOnes.data, Angle, ParaBack);
FidelityDenominator.data(FidelityDenominator.data == 0) = 1; 
        
%% %%%%%%%%%%%%%%%%%%%%%%%%% iterative reconstruction %%%%%%%%%%%%%%%%%%%%%%%%
Image = zeros(ParaBack(1), ParaBack(2));  
for id_iteration = 1 : iteration
    for id_subset = 1 : subset
        CurrentProjection =  ForwardProjection(Image, Angle(id_subset : subset : NumberofViews), ParaForward);
        ResProjection = (CurrentProjection.data - RealProjection(:, id_subset : subset : NumberofViews));
        CurrentImage = BackProjection(ResProjection, Angle(id_subset : subset : NumberofViews), ParaBack);
        Image = Image - (subset * CurrentImage.data ) ./ FidelityDenominator.data;
        Image(Image < 0) = 0;      
    end
end
output = Image;
end


