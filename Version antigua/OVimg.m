
function [OverlayImagen, alpha] = OVimg(Y, X, strengthPercent, height, width) %#ok<STOUT> 
    OverlayImage = [];
    F = scatteredInterpolant(Y, X, strengthPercent, 'linear');

    for i = 1:height-1
       for j = 1:width-1
              OverlayImage(i,j) = F(i,j); %#ok<AGROW> 
       end
    end

    alpha = (~isnan(OverlayImage)) * 0.4;
end

