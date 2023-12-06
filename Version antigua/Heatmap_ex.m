
strength = [10 20]';
X = [10 20]';
Y = [10 20]';
[OverlayImage, alpha] = OVimg(Y,X,strength, 500, 500);
h = imshow(OverlayImage);

colormap(h.Parent, jet);
colorbar(h.Parent);
set(h,'AlphaData',alpha); 