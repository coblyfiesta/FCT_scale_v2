pf_param = struct('affsig', [10,10,.004,.00,0.00,0], 'p_sz', 64,...
            'p_num', 600, 'mv_thr', 0.1, 'up_thr', 0.35, 'pos_thr', 0.3, 'roi_scale', 2); % roi_scale = 2;

        
scale_param.scale_sigma_factor = 1/4;        % standard deviation for the desired scale filter output
scale_param.lambda = 1e-2;					% regularization weight (denoted "lambda" in the paper)
scale_param.learning_rate = 0.025;%0.025;			% tracking model learning rate (denoted "eta" in the paper)
scale_param.number_of_scales = 33;%33;%55;%33;           % number of scale levels (denoted "S" in the paper)
scale_param.scale_step = 1.02;%1.1;%1.02;               % Scale increment factor (denoted "a" in the paper)
scale_param.scale_model_max_area = 512;      % the maximum size of scale examples       



close all;

track_res = ['benchmark_res/'];
if ~isdir(track_res)
    mkdir(track_res);
end
sample_res = ['sample_res/' path_name '/' set_name '/'];
if ~isdir(sample_res)
    mkdir(sample_res);
end

data_path = ['~/Downloads/imagedata++/' path_name '/' set_name '/'];
gt_path = ['~/Downloads/alov300++_rectangleAnnotation_full/' path_name '/' set_name '.ann'];
GT = load(gt_path);

init_box = [GT(1, 4:5), GT(1, 8:9)];
init_location = [init_box(1:2), init_box(3:4)-init_box(1:2)];
dia = (init_location(1, 3)^2+init_location(1, 4)^2)^0.5;

% scale = gt(1, 3)/ gt (1, 4);
scale = [dia/init_location(1, 3), dia/init_location(1, 4)];
l1_off = [0,0];
l2_off = [0,0];
s1 = pf_param.roi_scale*[scale(1),scale(2)];
s2 = pf_param.roi_scale*[scale(1),scale(2)];

feature_solver_def_file = 'feature_solver.prototxt';

gnet_solver_def_file = ['solver/gnet_solver_' num2str(ch_num) '.prototxt'];
lnet_solver_def_file = ['solver/lnet_solver_' num2str(ch_num) '.prototxt']; 
select_lnet_solver_def_file = 'select_lnet_solver.prototxt'; 
select_gnet_solver_def_file = 'select_gnet_solver.prototxt';
model_file = 'VGG_ILSVRC_16_layers.caffemodel';
fsolver = caffe.Solver(feature_solver_def_file);
fsolver.net.copy_from(model_file);
gsolver = caffe.Solver(gnet_solver_def_file);
lsolver = caffe.Solver(lnet_solver_def_file);

% caffe('init_solver', feature_solver_def_file, model_file);
% caffe('init_gsolver', select_gnet_solver_def_file, model_file);
% caffe('init_lsolver', select_lnet_solver_def_file, model_file);
% caffe('set_mode_gpu');
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

roi_size = 361;%368; %380;

mean_pix = [103.939, 116.779, 123.68]; 

% fnum = 20;
location = init_location;
% location = GT(1,:);
pf_param.ratio = location(3)/pf_param.p_sz;
pf_param.affsig(3) = pf_param.affsig(3)*pf_param.ratio;
pf_param.affsig_o = pf_param.affsig;
pf_param.affsig(3) = 0;
pf_param.minconf = 0.5;
