function[scale_param] = init_scale_estimator(im, location, scale_param)

% desired scale filter output (gaussian shaped), bandwidth proportional to
% number of scales   

init_target_sz = location([3,4]);

scale_param.scale_sigma = scale_param.number_of_scales/sqrt(33) * scale_param.scale_sigma_factor;
ss = (1:scale_param.number_of_scales) - ceil(scale_param.number_of_scales/2);
ys = exp(-0.5 * (ss.^2) / scale_param.scale_sigma^2);
scale_param.ysf = single(fft(ys));


% store pre-computed scale filter cosine window
if mod(scale_param.number_of_scales,2) == 0
    scale_param.scale_window = single(hann(scale_param.number_of_scales+1));
    scale_param.scale_window = scale_param.scale_window(2:end);
else
    scale_param.scale_window = single(hann(scale_param.number_of_scales));
end;

% scale factors
ss = 1:scale_param.number_of_scales;
scale_param.scaleFactors = scale_param.scale_step.^(ceil(scale_param.number_of_scales/2) - ss);

% compute the resize dimensions used for feature extraction in the scale
% estimation
scale_param.scale_model_factor = 1;
if prod(init_target_sz) > scale_param.scale_model_max_area
    scale_param.scale_model_factor = sqrt(scale_param.scale_model_max_area/prod(init_target_sz));
end
scale_param.scale_model_sz = floor(init_target_sz * scale_param.scale_model_factor);

scale_param.currentScaleFactor = 1;

% scale_param.min_scale_factor = scale_param.scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_param.scale_step));
% scale_param.max_scale_factor = scale_param.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));

%% extract the training sample feature map for the scale filter
cur_location = [location(1) + floor(location(3)/2), location(2) + floor(location(4)/2), location(3), location(4)]; % [cx, cy, w, h]
xs = get_scale_sample(im, cur_location, scale_param.currentScaleFactor * scale_param.scaleFactors, scale_param.scale_window, scale_param.scale_model_sz);

%% calculate the scale filter update
xsf = fft(xs,[],2);
new_sf_num = bsxfun(@times, scale_param.ysf, conj(xsf));
new_sf_den = sum(xsf .* conj(xsf), 1);

scale_param.sf_den = new_sf_den;
scale_param.sf_num = new_sf_num;
end

