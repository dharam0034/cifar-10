function [f]= cifar_10_features(x)
    f=[mean(x(1:1024)) mean(x(1025:2048)) mean(x(2049:3072))];
end