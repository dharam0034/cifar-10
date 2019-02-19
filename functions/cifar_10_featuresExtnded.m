function [f]= cifar_10_featuresExtnded(x,N)
    
    data_sample = x;
    img_r = data_sample(1:1024);
    img_g = data_sample(1025:2048);
    img_b = data_sample(2049:3072);
    data_img = zeros(32,32,3);
    data_img(:,:,1) = reshape(img_r, [32 32])';
    data_img(:,:,2) = reshape(img_g, [32 32])';
    data_img(:,:,3) = reshape(img_b, [32 32])';    

    x=ones(1,32/N).*N;
    y=ones(1,32/N).*N;
    img_cell=mat2cell(data_img,x,y,[3]);
    cell_v = img_cell(:);
        
    f=[];
    for j=1:length(cell_v)
        f=[f mean2(cell_v{j}(:,:,1)) mean2(cell_v{j}(:,:,2)) mean2(cell_v{j}(:,:,3))];
    end 
end