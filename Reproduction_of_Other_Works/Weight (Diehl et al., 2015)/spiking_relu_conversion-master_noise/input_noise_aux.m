load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

mask = (rand(size(test_x)) - 0.5) .* 2;
test_x = test_x .* (1 + mask); 


load mnist_uint8;
train_x = double(reshape(train_x',28,28,60000)) / 255;
test_x = double(reshape(test_x',28,28,10000)) / 255;
train_y = double(train_y');
test_y = double(test_y');

mask = (rand(size(test_x)) - 0.5) .* 0.6;
test_x = test_x .* (1 + mask); 