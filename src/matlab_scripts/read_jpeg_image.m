function [original_dct, quant_table, spatial_image] = read_jpeg_image(image_file_path)

image_struct = jpeg_read(image_file_path);
original_dct = PlaneToVec(image_struct.coef_arrays{1});
quant_table = image_struct.quant_tables{image_struct.comp_info(1).quant_tbl_no};
spatial_image = 1;
% spatial_image = double(Dct2Spat(original_dct, quant_table));

end

function plane = VecToPlane(Cube)
[Y, X, ~] = size(Cube);
M = floor(X * 8);                   % count of columns
N = floor(Y * 8);                   % count of rows
plane = zeros([N, M]);
for i = 1 : Y
    for j = 1 : X
        plane(((i-1)*8+1):i*8, ((j-1)*8+1):j*8) = reshape(Cube(i,j,:),8,8);
    end
end

end

function Mat = PlaneToVec(plane)
[Y, X] = size(plane);
M = floor(X / 8);	                % count of columns
N = floor(Y / 8);  	                % count of rows
Mat = zeros(N, M, 64);
for i = 1 : N
    for j = 1 : M
        Mat(i,j,:) = reshape(plane(((i-1)*8+1):i*8, ((j-1)*8+1):j*8), 64, 1);
    end
end

end

function op = Dct2Spat(X, Q)
Step1 = VecToPlane(X);
op = blkproc(Step1, [8, 8], @jpg2raw02,Q);
op = uint8(op);

end

function X = jpg2raw02(QD, Qf)

% "Inverse" function to raw2jpg02.m, calculates the spatial representation for an 8x8 block
% of DCT coefficients QD quantized with matrix Qf
%
% Input:
% QD = Matrix of DCT coefficients quantized with the quantization matrix Qf (QD is 8x8 of integers)
% Qf = Quantization matrix (8x8 of quantization factors)
%
% Output:
% X  = Spatial representation of QD = DCT^(-1)(QD) rounded to integers and at 0 and 255 

% 128 is added because in calculating the DCT QD coeffs, 128 is subtracted (see raw2jpg01.m)

X = 128 + round(idct2(QD.*Qf));     % Rounding to integers
X(X < 0) = 0;                       % Truncating at 0
X(X > 255) = 255;                   % Truncating at 255

end