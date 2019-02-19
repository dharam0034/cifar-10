function [classes]=cifar_10_MLP_test(x,net) 

    y = net(x);
    classes = vec2ind(y);

end 
