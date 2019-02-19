%%This intial code block was provided by professor

if exist('tr_data','var') && size(tr_data,1) == 50000
  disp('Seems that data exists, clean tr_data to re-read!');
  return;
end;

conf.cifar10_dir = 'cifar-10-batches-mat';
conf.train_files = {'data_batch_1.mat',...
                    'data_batch_2.mat',...
                    'data_batch_3.mat',...
                    'data_batch_4.mat',...
                    'data_batch_5.mat'};
conf.test_file = 'test_batch.mat';
conf.meta_file = 'batches.meta.mat';

load(fullfile(conf.cifar10_dir,conf.meta_file));

% Read training data and form the feature matrix and target output
tr_data = [];
tr_labels = [];
fprintf('Reading training data...\n');
for train_file_ind = 1:length(conf.train_files)
  fprintf('\r  Reading %s', conf.train_files{train_file_ind});
  load(fullfile(conf.cifar10_dir,conf.train_files{train_file_ind}));
  tr_data = [tr_data; data];
  tr_labels = [tr_labels; labels];
end;
fprintf('Done!\n');

% Plot random figures 32x32=1024 pixels r,g,b channels
fprintf('Showing training data...\n');
for data_ind = 1:size(tr_data,1)
  if rand() < 0.0005
    data_sample = tr_data(data_ind,:);
    img_r = data_sample(1:1024);
    img_g = data_sample(1025:2048);
    img_b = data_sample(2049:3072);
    data_img = zeros(32,32,3);
    data_img(:,:,1) = reshape(img_r, [32 32])';
    data_img(:,:,2) = reshape(img_g, [32 32])';
    data_img(:,:,3) = reshape(img_b, [32 32])';
    imshow(data_img./256);
    title(label_names(tr_labels(data_ind)+1));
    drawnow;
%    pause(1);
    %input('  Training example <PRESS RETURN>')
  end;
end;
fprintf('Done!\n');

% Read test data and form the feature matrix and target output
fprintf('Reading and showing test data...\n');
load(fullfile(conf.cifar10_dir,conf.test_file));
te_data = data;
te_labels = labels;

for data_ind = 1:size(te_data,1)
  if rand() < 0.0005
    data_sample = te_data(data_ind,:);
    img_r = data_sample(1:1024);
    img_g = data_sample(1025:2048);
    img_b = data_sample(2049:3072);
    data_img = zeros(32,32,3);
    data_img(:,:,1) = reshape(img_r, [32 32])';
    data_img(:,:,2) = reshape(img_g, [32 32])';
    data_img(:,:,3) = reshape(img_b, [32 32])';
    imshow(data_img./256);
    title(label_names(te_labels(data_ind)+1));
    drawnow;
    %pause(1);
    %input('  Testing example <PRESS RETURN>')
  end;
end;
fprintf('Done!\n');

%% The code of different classifiers starts  here. 

%% random classifier 
predLablesRAND=[];
for i=1:length(te_data)
    predLablesRAND=[predLablesRAND,cifar_10_rand(te_data(i,:))];
end
predLablesRAND=predLablesRAND.';
cifar_10_evaluate(predLablesRAND, labels)

%% Nearest Neighbour classifier 
predLablesNN=[];
for i=1:length(te_data)
    predLablesNN=[predLablesNN,cifar_10_1NN(te_data(i,:),tr_data,tr_labels)];
    i
end
predLablesNN=predLablesNN.';
cifar_10_evaluate(predLablesNN, labels)

%% EX3(1), Naive Bayesian classification

% feature extraction 
tr_features=[];
te_features=[];
for i=1:length(tr_data)
    tr_features=[tr_features; cifar_10_features(tr_data(i,:))];
end
for i=1:length(te_data)
    te_features=[te_features; cifar_10_features(te_data(i,:))];
end

% learn var
[mu sigma p] = cifar_10_bayes_learn(tr_features,tr_labels);

% classify/ pridict
predLablesBC=[];
for i=1:length(te_features)
    predLablesBC=[predLablesBC,cifar_10_bayes_classify(te_features(i,:),mu,sigma,p)];
end


predLablesBC=predLablesBC.';
cifar_10_evaluate(predLablesBC, labels)

%% EX3(2) Bayesian classification with multivariate normal distribution

%feature extraction 
tr_features=[];
te_features=[];
for i=1:length(tr_data)
    tr_features=[tr_features; cifar_10_features(tr_data(i,:))];
end
for i=1:length(te_data)
    te_features=[te_features; cifar_10_features(te_data(i,:))];
end


%learn cov
[mu Sigma p] = cifar_10_bayes_learn_1(tr_features,tr_labels);
%classify/pridict
predLablesBC=[];
for i=1:length(te_features)
    predLablesBC=[predLablesBC,cifar_10_bayes_classify_1(te_features(i,:),mu,Sigma,p)];
end
predLablesBC=predLablesBC.';
cifar_10_evaluate(predLablesBC, labels)

%% EX3(3), Bayesian with extended features 

%feature extraction 
% N= sub-block size N*N*3
subBlockSizes=[32 16 8 4 2];
subBlockNames={'features_32', 'features_16', 'features_8','features_4','features_2'};

for k=1:length(subBlockSizes)-1
    N=subBlockSizes(k)
    tr_features_temp=[];te_features_temp=[];
    for i=1:length(tr_data)
        i
        tr_features_temp=[tr_features_temp; cifar_10_featuresExtnded(tr_data(i,:),N)];
    end
    for i=1:length(te_data)
        i
        te_features_temp=[te_features_temp; cifar_10_featuresExtnded(te_data(i,:),N)];
    end
    k
    featuresAll(k).name=subBlockNames(k);
    featuresAll(k).tr_features=tr_features_temp;
    featuresAll(k).te_features=te_features_temp;
end 

predSubblocks=[];
for j=1:(length(featuresAll))
    j
    %learn cov
    [mu_Ex, Sigma_Ex, p_Ex] = cifar_10_bayes_learnEx(featuresAll(j).tr_features,tr_labels);
    %classify/pridict    
    predLablesBC=[];
    for i=1:1:length(featuresAll(j).te_features)
        temp=cifar_10_bayes_classifyEx(featuresAll(j).te_features(i,:),mu_Ex,Sigma_Ex,p_Ex);
        predLablesBC=[predLablesBC, temp(1)];
    end
    predLablesBC=predLablesBC.';
    predSubblocks=[predSubblocks cifar_10_evaluate(predLablesBC, labels)];
end
plot(predSubblocks)



%% EX4, MLP classifier
N=32;
tr_features_32=zeros(length(tr_data),3*1);
te_features_32=zeros(length(te_data),3*1);
for i=1:length(tr_data)
    i
    tr_features_32(i,:)=cifar_10_featuresExtnded(tr_data(i,:),N);
end
for i=1:length(te_data)
    i
    te_features_32(i,:)=cifar_10_featuresExtnded(te_data(i,:),N);
end

N=16;
tr_features_16=zeros(length(tr_data),3*4);
te_features_16=zeros(length(te_data),3*4);
for i=1:length(tr_data)
    i
    tr_features_16(i,:)=cifar_10_featuresExtnded(tr_data(i,:),N);
end
for i=1:length(te_data)
    i
    te_features_16(i,:)=cifar_10_featuresExtnded(te_data(i,:),N);
end

N=8;
tr_features_8=zeros(length(tr_data),3*16);
te_features_8=zeros(length(te_data),3*16);
for i=1:length(tr_data)
    i
    tr_features_8(i,:)=cifar_10_featuresExtnded(tr_data(i,:),N);
end
for i=1:length(te_data)
    i
    te_features_8(i,:)=cifar_10_featuresExtnded(te_data(i,:),N);
end

tr_features_32_1= tr_features_32.';te_features_32_1= te_features_32.';
tr_features_16_1= tr_features_16.';te_features_16_1= te_features_16.';
tr_features_8_1= tr_features_8.';te_features_8_1= te_features_8.';

tr_l=zeros(1,50000);te_l=zeros(1,10000);

tr_labels = tr_labels.';te_labels = te_labels.';

label_no=unique(tr_labels,'sorted')
id_trl = arrayfun( @(x)( find(tr_labels==x) ), label_no,'UniformOutput',false );
id_tel = arrayfun( @(x)( find(te_labels==x) ), label_no,'UniformOutput',false );

tr_labels_one_hot = zeros( 10 , length( tr_labels ) );
te_labels_one_hot = zeros( 10 , length( te_labels ) );

for k=1:10
    for m=1:length(id_tel{k})
        te_labels_one_hot(k,id_tel{k}(m))=1;
    end
end
for k=1:10
    for m=1:length(id_trl{k})
        tr_labels_one_hot(k,id_trl{k}(m))=1;
    end     
end

net=cifar_10_MLP_train(tr_features_8_1,tr_labels_one_hot)
[classes]=cifar_10_MLP_test(te_features_8_1,net) 
classes = classes -1;

predLablesPR=classes.';
cifar_10_evaluate(predLablesPR, labels)

% featuresAll(1).name='features_32';
% featuresAll(1).tr_features=tr_features_32;
% featuresAll(1).te_features=te_features_32;
% featuresAll(2).name='features_16';
% featuresAll(2).tr_features=tr_features_16;
% featuresAll(2).te_features=te_features_16;
% featuresAll(3).name='features_8';
% featuresAll(3).tr_features=tr_features_8;
% featuresAll(3).te_features=te_features_8;
% featuresAll(4).name='features_4';
% featuresAll(4).tr_features=tr_features_4;
% featuresAll(4).te_features=te_features_4;
% featuresAll(5).name='features_2';
% featuresAll(5).tr_features=tr_features_2;
% featuresAll(5).te_features=te_features_2;
% 

N=32;
tr_features_32=[];te_features_32=[];
for i=1:length(tr_data)
    i
    tr_features_32=[tr_features_32; cifar_10_featuresExtnded(tr_data(i,:),N)];
end
for i=1:length(te_data)
    i
    te_features_32=[te_features_32; cifar_10_featuresExtnded(te_data(i,:),N)];
end

% N=16;
% tr_features_16=[];te_features_16=[];
% for i=1:length(tr_data)
%     i
%     tr_features_16=[tr_features_16; cifar_10_featuresExtnded(tr_data(i,:),N)];
% end
% for i=1:length(te_data)
%     i
%     te_features_16=[te_features_16; cifar_10_featuresExtnded(te_data(i,:),N)];
% end
% 
% 
% N=8;
% tr_features_8=[];te_features_8=[];
% for i=1:length(tr_data)
%     i
%     tr_features_8=[tr_features_8; cifar_10_featuresExtnded(tr_data(i,:),N)];
% end
% for i=1:length(te_data)
%     i
%     te_features_8=[te_features_8; cifar_10_featuresExtnded(te_data(i,:),N)];
% end
% 
% N=4;
% tr_features_4=[];te_features_4=[];
% for i=1:length(tr_data)
%     i
%     tr_features_4=[tr_features_4; cifar_10_featuresExtnded(tr_data(i,:),N)];
% end
% for i=1:length(te_data)
%     i
%     te_features_4=[te_features_4; cifar_10_featuresExtnded(te_data(i,:),N)];
% end
% 
% N=2;
% tr_features_2=[];te_features_2=[];
% for i=1:length(tr_data)
%     %i
%     tr_features_2=[tr_features_2; cifar_10_featuresExtnded(tr_data(i,:),N)];
% end
% for i=1:length(te_data)
%     %i
%     te_features_2=[te_features_2; cifar_10_featuresExtnded(te_data(i,:),N)];
% end