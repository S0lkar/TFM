% Enter data by hand if data from a ThingSpeak channel is not available.
strength = [-90 -90 -90 -90]';
% Read data from a ThingSpeak channel.
% Uncomment the next line to read from ThingSpeak.
% strength = thingSpeakRead(CHANNEL_ID, ReadKey',READ_API_KEY,'numPoints',15,'fields',FIELD_NUMBER');
X = [10 550 550 10]';
Y = [10 410 400 293]';
strengthPercent = 2*(strength+100)/100;

[I,m] = imread('https://www.mathworks.com/help/examples/thingspeak/win64/CreateHeatmapOverlayImageTSExample_02.png','png');
picture=ind2rgb(I,m); 
[height,width,depth] = size(picture); 

OverlayImage=[];
F = scatteredInterpolant(Y, X, strengthPercent,'linear');
for i = 1:height-1
   for j = 1:width-1
          OverlayImage(i,j) = F(i,j);
   end
end
alpha = (~isnan(OverlayImage))*0.4;

imshow(picture,m);
hold on

h = imshow(OverlayImage);

colormap(h.Parent, jet);
colorbar(h.Parent);
set(h,'AlphaData',alpha); 