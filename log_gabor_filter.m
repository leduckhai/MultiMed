%   Log-Gabor Filter function
%
%   Le Duc Khai
%   Bachelor in Biomedical Engineering
%   FH Aachen - University of Applied Sciences, Germany.
%
%   Last updated on 11.02.2019.
%
%   The proposed algorithm creates log-Gabor Filters, which is often used
%   to denoise images.
%
%   Implementation is based on this scientific paper:
%       Fischer, S & Redondo, Rafael & Cristobal, Gabriel
%       "How to construct log-Gabor Filters?"
%   
%   The following codes are implemented only for PERSONAL USE, e.g improving
%   programming skills in the domain of Image Processing and Computer Vision.
%   If you use this algorithm, please cite the paper mentioned above to support
%   the authors.
%
%   Parameters:
%       image: the input image 
%       scale: number of wavelet scales
%       orientation: number of orientations
%       min_wavelength: wavelength of smallest scale filter
%       factor: scaling factor between successive filters.
%       sigmaOnf: Ratio of the standard deviation of the Gaussian describing
%                 the log Gabor filter's transfer function in the frequency 
%                 domain to the filter center frequency.
%       dThetaOnSigma: Ratio of angular interval between filter orientations
%                      and the standard deviation of the angular Gaussian
%                      function used to construct filters in the frequency
%                      plane.
%                                               
%   Example use: 
%       scale =4; 
%       orientation = 6;
%       min_wavelength = 3;
%       factor = 1.7;
%       sigmaOnf = 0.65;
%       dThetaOnSigma = 1.3;
%       result_image = log_gabor_filter(I, 4, 6, 3, 1.7, 0.65, 1.3)
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
function result_image = log_gabor_filter(image, scale, orientation, min_wavelength, factor, ...
			    sigmaOnf, dThetaOnSigma)

% Gray scale image converted
if ndims(image) == 3
    disp('Color image is being converted into gray scale image');
    image = rgb2gray(image);
end

% Fourier transform of the image
[rows cols] = size(image);
imagefft = fft2(image); 

% Create 2 matrices X and Y with ranges normalised to +/- 0.5. All elements of x have a value equal to its 
% x coordinate relative to the centre, elements of y have values equal to 
% their y coordinate relative to the centre.
x = ones(rows,1) * (-cols/2 : (cols/2 - 1))/cols; 
y = (-rows/2 : (rows/2 - 1))' * ones(1,cols)/rows;    
radius = sqrt(x.^2 + y.^2);       
radius(rows/2+1, cols/2+1) = 1;     % Remove radius = 0 for possible log-function
theta = atan2(y,x);     % Angular distance between x and y  
sintheta = sin(theta);
costheta = cos(theta);          

% Log Gabor function
for i = 1:scale
    wavelength = min_wavelength*factor^(i-1);
    fo = 1.0/wavelength;    % Centre frequency of filter.
    logGabor = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2)); 
    logGabor(round(rows/2+1),round(cols/2+1)) = 0; % At the 0-frequency, log function is 0
end

% Log Gabor filter
for j = 1:orientation
    angle = (j-1)*pi/orientation;   % Calculate filter angle
    wavelength = min_wavelength;    % Initial filter wavelength        
    dsin = sintheta * cos(angle) - costheta * sin(angle);     % Difference in sine
    dcos = costheta * cos(angle) + sintheta * sin(angle);     % Difference in cosine
    dtheta = abs(atan2(dsin,dcos));                           % Absolute angular distance.
    thetaSigma = pi/orientation/dThetaOnSigma; % Standard deviation of the angular Gaussian function
                                               % used to construct filters in the frequency plane.
    spread = exp((-dtheta.^2) / (2 * thetaSigma^2));  
    for i = 1:scale
        filter = fftshift(logGabor .* spread);
        result_image = ifft2(imagefft .* filter);
        result_image = real(result_image); % Only take real parts of the image
        wavelength = wavelength * factor; % Calculate wavelength of next filter
    end       
end  
    
