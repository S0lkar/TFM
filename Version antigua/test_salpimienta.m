
imagen= rgb2gray(imread('test.png'));
% Aplicar un filtro de suavizado
% Realzado con Roberts
umbral = 0.2;
hf=[0 0 0; 0 0 1; 0 -1 0];
hc=[-1 0 0; 0 1 0; 0 0 0];
Gx= imfilter(double(imagen),hf,'conv'); % Gradiente fila
Gy= imfilter(double(imagen),hc,'conv'); % Gradiente columna
G= sqrt(Gx.*Gx+Gy.*Gy); % magnitud
% Fijar un valor de umbral
I = G > umbral;

% I = medfilt2(imagen, [5,5]); hace la cosa por erosión



figure, imshow(I, [])
figure, imshow(imagen, [])
%save('res_test.png', 'I')