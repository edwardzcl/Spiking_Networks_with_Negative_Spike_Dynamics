% TD = Read_Ndataset(filename)
% returns the Temporal Difference (TD) events from binary file for the
% N-MNIST and N-Caltech101 datasets. See garrickorchard.com\datasets for
% more info
function TD = Read_Ndataset(filename)
eventData = fopen(filename);
evtStream = fread(eventData);
fclose(eventData);

TD.x    = evtStream(1:5:end)+1; %pixel x address, with first pixel having index 1
TD.y    = evtStream(2:5:end)+1; %pixel y address, with first pixel having index 1
TD.p    = bitshift(evtStream(3:5:end), -7)+1; %polarity, 1 means off, 2 means on
TD.ts   = bitshift(bitand(evtStream(3:5:end), 127), 16); %time in microseconds
TD.ts   = TD.ts + bitshift(evtStream(4:5:end), 8);
TD.ts   = TD.ts + evtStream(5:5:end);
return

eventData = fopen("00003.bin");
evtStream = fread(eventData);
fclose(eventData);

TD.x    = evtStream(1:5:end)+1; %pixel x address, with first pixel having index 1
TD.y    = evtStream(2:5:end)+1; %pixel y address, with first pixel having index 1
TD.p    = bitshift(evtStream(3:5:end), -7)+1; %polarity, 1 means off, 2 means on
TD.ts   = bitshift(bitand(evtStream(3:5:end), 127), 16); %time in microseconds
TD.ts   = TD.ts + bitshift(evtStream(4:5:end), 8);
TD.ts   = TD.ts + evtStream(5:5:end);

mnist=zeros(34,34);
for i=1:length(TD.ts)
    if TD.p(i)==1
        mnist(TD.x(i),TD.y(i))=mnist(TD.x(i),TD.y(i))+0;
    else
        if TD.p(i)==2
            mnist(TD.x(i),TD.y(i))=mnist(TD.x(i),TD.y(i))+1;
        end
    end
end
imshow(mnist)
            
        
