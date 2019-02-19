function mat = label2mat(label, size)

if ~exist('size', 'var') || isempty(size)
    size = 10;
end

if label > size
    error('Label (%d) should be < size (%d).', label, size);
end

I = eye(size);
mat = I(:, label);

end