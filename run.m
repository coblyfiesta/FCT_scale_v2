
data_path = '~/Downloads/imagedata++/';
dataset1 = dir(data_path);
%
for i=1:length(dataset1)
    if ~isdir([data_path dataset1(i).name]) || strcmp(dataset1(i).name, '.') || strcmp(dataset1(i).name, '..') || strcmp(dataset1(i).name, 'AE_train_Deer')
        continue;
    end
    dataset2 = dir([data_path dataset1(i).name]);
    for j = 1:length(dataset2)
        if ~isdir([data_path dataset1(i).name '/' dataset2(j).name]) || strcmp(dataset2(j).name, '.') || strcmp(dataset2(j).name, '..')
            continue;
        end
        cnn2_pf_tracker(dataset1(i).name, dataset2(j).name, 1, 512);
    end
end

% for i=31:length(dataset)
%     if ~isdir([data_path dataset(i).name]) || strcmp(dataset(i).name, '.') || strcmp(dataset(i).name, '..') || strcmp(dataset(i).name, 'AE_train_Deer')
%         continue;
%     end
%     cnn2_pf_tracker(dataset(i).name, 1, 512);
% % end
% for i=51:length(dataset)
%     if ~isdir([data_path dataset(i).name]) || strcmp(dataset(i).name, '.') || strcmp(dataset(i).name, '..') || strcmp(dataset(i).name, 'AE_train_Deer')
%         continue;
%     end
%     cnn2_pf_tracker(dataset(i).name, 1, 256);
% end
% 
% 
% 
% 
% 
% for i=1:length(dataset)
%     if ~isdir([data_path dataset(i).name]) || strcmp(dataset(i).name, '.') || strcmp(dataset(i).name, '..') || strcmp(dataset(i).name, 'AE_train_Deer')
%         continue;
%     end
%     cnn2_pf_tracker(dataset(i).name, 1, 128);
% end
% 
% for i=1:length(dataset)
%     if ~isdir([data_path dataset(i).name]) || strcmp(dataset(i).name, '.') || strcmp(dataset(i).name, '..') || strcmp(dataset(i).name, 'AE_train_Deer')
%         continue;
%     end
%     cnn2_pf_tracker(dataset(i).name, 1, 64);
% end% 
% % iter = 50;
% for i=1:iter
%     l_pre_map = caffe('forward_lnet', {lfea2_train});
%     diff{1}(:,:,:,1) = 0.5*(l_pre_map{1}(:,:,:,1)-permute(single(map2_store), [2,1,3]));
%     diff{1}(:,:,:,2) = 0.3*squeeze(l_pre_map{1}(:,:,:,2)-permute(single(map),[2,1,3])).*permute(single(map<=0), [2,1,3]);
%     caffe('backward_lnet', diff);
%     caffe('update_lnet');
%     %                     l_pre_map = caffe('forward_lnet', {lfea2});
%     %                     %         diff = permute(l_pre_map-single(map), [2,1,3]);
%     %                     %                     diff = l_pre_map{1}-single(l_pre_map_o{1}>0.01).*single(permute(map>0, [2,1,3]));
%     %                     diff = (l_pre_map-single(map)).*permute(single(map<=0), [2,1,3]);
%     %                     caffe('backward_lnet', {diff});
%     %                     caffe('update_lnet');
%     figure(50); subplot(1,2,1); imagesc(permute(l_pre_map{1}(:,:,:,1), [2,1,3]));
%     figure(50); subplot(1,2,2); imagesc(permute(l_pre_map{1}(:,:,:,2), [2,1,3]));
% end