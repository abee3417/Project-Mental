raw = readtable('raw_data1.csv');
raw2 = readtable('raw_data2.csv');
raw2.reg_date = erase(raw2.reg_date, "'");
%raw.reg_date = datetime(raw.reg_date);
%raw2.reg_date = datetime(raw2.reg_date);
raw = sortrows(raw, "reg_date");
raw2 = sortrows(raw2, "reg_date");
id_list = unique(raw.menti_seq);

x0_cell = {};
x1_cell = {};
x2_cell = {};
x0_arr = [];
x1_arr = [];
x2_arr = [];
y0_arr = [];
y1_arr = [];
y2_arr = [];


i0 = 1;
i1 = 1;
i2 = 1;
emp = {0, 0, 0, 0, 0, 0};

for i=1:length(id_list)
    id = id_list(i);
    y0 = raw2((raw2.menti_seq == id) & (raw2.srvy_name=="PHQ-9"), 3).srvy_result;
    y1 = raw2((raw2.menti_seq == id) & (raw2.srvy_name=="P4"), 3).srvy_result;
    y2 = raw2((raw2.menti_seq == id) & (raw2.srvy_name=="Loneliness"), 3).srvy_result;
    r = raw(raw.menti_seq == id, 5:10);
    % r = [repmat(emp,40-height(r),1);r];
    r = r(end, :);
    if ~isempty(y0)
        k = y0(end) == 0;
        y0_arr = [y0_arr; k];
        x0_cell{i0, 1} = table2array(r)';
        x0_arr = [x0_arr; table2array(r)];
        i0 = i0 + 1;
    end
    if ~isempty(y1)
        k = y1(end) == 0;
        y1_arr = [y1_arr; k];
        x1_cell{i1, 1} = table2array(r)';
        x1_arr = [x1_arr; table2array(r)];
        i1 = i1 + 1;
    end
    if ~isempty(y2)
        k = y2(end) == 0;
        y2_arr = [y2_arr; k];
        x2_cell{i2, 1} = table2array(r)';
        x2_arr = [x2_arr; table2array(r)];
        i2 = i2 + 1;
    end

end

mode = "P4";

if mode == "PHQ-9"
    target_x = x0_cell;
    target_y = y0_arr;
elseif mode == "P4"
    target_x = x1_cell;
    target_y = y1_arr;
else
    target_x = x2_cell;
    target_y = y2_arr;
end

train_size = round(length(target_x(:,1)) * 0.8);
val_size = length(target_x(:,1)) - train_size;
target_y = categorical(target_y);
XTrain = target_x(1:train_size, :);
TTrain = target_y(1:train_size, :);
XTest = target_x(train_size+1:end, :);
TTest = target_y(train_size+1:end, :);

layers = [
    sequenceInputLayer(6)
    lstmLayer(2, OutputMode='last')
    % fullyConnectedLayer(2)
    % reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions("adam", ...
    'ValidationData', {XTest, TTest}, ...
    MaxEpochs=20000, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    InitialLearnRate=1e-3, ...
    MiniBatchSize=2048, ...
    OutputNetwork='best-validation-loss', ...
    ValidationFrequency=20, ...
    ValidationPatience=300, ...
    Verbose=0);

[net, info] = trainNetwork(XTrain,TTrain,layers,options);
% test on validation images
Pred = classify(net, XTest);
%Pred = predict(net, XTest);
cm = confusionchart(TTest, Pred)
%plot([Pred, TTest])



