clear;
clc;
% Your path
train_path = dir(fullfile('F:', 'matlab', 'cnn_snn_simulator', 'N-MNIST', 'Train'));
test_path = dir(fullfile('F:', 'matlab', 'cnn_snn_simulator', 'N-MNIST', 'Test'));

n_mnist.train_data = zeros(34,34,60000);
n_mnist.train_label = zeros(1,60000);
n_mnist.test_data = zeros(34,34,10000);
n_mnist.test_label = zeros(1,10000);


train_num = 1;
for i=3:length(train_path)
    disp(['processing train category: ', num2str(i-3)]);
    category{i-2} = dir(fullfile(train_path(i).folder, train_path(i).name));
    for j=3:length(category{i-2})
        eventData = fopen(fullfile(category{i-2}(j).folder, category{i-2}(j).name));
        evtStream = fread(eventData);
        fclose(eventData);
        TD.x    = evtStream(1:5:end)+1; %pixel x address, with first pixel having index 1
        TD.y    = evtStream(2:5:end)+1; %pixel y address, with first pixel having index 1
        TD.p    = bitshift(evtStream(3:5:end), -7)+1; %polarity, 1 means off, 2 means on
        TD.ts   = bitshift(bitand(evtStream(3:5:end), 127), 16); %time in microseconds
        TD.ts   = TD.ts + bitshift(evtStream(4:5:end), 8);
        TD.ts   = TD.ts + evtStream(5:5:end);
        
        train_data = zeros(34,34);
        for k=1:length(TD.ts)
            if TD.p(k)==1
                train_data(TD.x(k),TD.y(k))=train_data(TD.x(k),TD.y(k))+0;
            elseif TD.p(k)==2
                train_data(TD.x(k),TD.y(k))=train_data(TD.x(k),TD.y(k))+1;
            end
        end
        %n_mnist.train_data(TD.x(k),TD.y(k),end+1)=train_data(TD.x(k),TD.y(k),:);
        n_mnist.train_data(:,:,train_num)=train_data(:,:);
        n_mnist.train_label(1,train_num) = i-3;
        train_num = train_num + 1;
    end
    %n_mnist.train_label() = ones(1,length(category{i-2}))*(i-3);
end

%n_mnist.train_data = n_mnist.train_data(:,:,2:end);

test_num = 1;
for i=3:length(test_path)
    disp(['processing test category: ', num2str(i-3)]);
    category{i-2} = dir(fullfile(test_path(i).folder, test_path(i).name));
    for j=3:length(category{i-2})
        eventData = fopen(fullfile(category{i-2}(j).folder, category{i-2}(j).name));
        evtStream = fread(eventData);
        fclose(eventData);
        TD.x    = evtStream(1:5:end)+1; %pixel x address, with first pixel having index 1
        TD.y    = evtStream(2:5:end)+1; %pixel y address, with first pixel having index 1
        TD.p    = bitshift(evtStream(3:5:end), -7)+1; %polarity, 1 means off, 2 means on
        TD.ts   = bitshift(bitand(evtStream(3:5:end), 127), 16); %time in microseconds
        TD.ts   = TD.ts + bitshift(evtStream(4:5:end), 8);
        TD.ts   = TD.ts + evtStream(5:5:end);
        
        test_data = zeros(34,34);
        for k=1:length(TD.ts)
            if TD.p(k)==1
                test_data(TD.x(k),TD.y(k))=test_data(TD.x(k),TD.y(k))+0;
            elseif TD.p(k)==2
                test_data(TD.x(k),TD.y(k))=test_data(TD.x(k),TD.y(k))+1;
            end
        end
        %n_mnist.test_data(TD.x(k),TD.y(k),end+1)=n_mnist.test_data(TD.x(k),TD.y(k))+1;
        %end+1的三种情况
        n_mnist.test_data(:,:,test_num)=test_data(:,:);
        n_mnist.test_label(1,test_num) = i-3;
        test_num = test_num + 1;
    end
    %n_mnist.test_label = ones(1,length(category{i-2}))*(i-3);
end

disp(['processing completed!!!']);


%n_mnist.train_data(find(n_mnist.train_data(:,:,:) > 30)) = 30; 
%n_mnist.test_data(find(n_mnist.test_data(:,:,:) > 30)) = 30; 
%n_mnist.test_data = n_mnist.test_data(:,:,2:end);
%shuffle
r = randperm(size(n_mnist.train_data,3));
n_mnist.train_data = n_mnist.train_data(:,:,r);
n_mnist.train_label = n_mnist.train_label(1,r);

r = randperm(size(n_mnist.test_data,3));
n_mnist.test_data = n_mnist.test_data(:,:,r);
n_mnist.test_label = n_mnist.test_label(1,r);

disp('Show a test sample: ');
imshow(n_mnist.test_data(:,:,1)'./max(max(n_mnist.test_data(:,:,1))));
