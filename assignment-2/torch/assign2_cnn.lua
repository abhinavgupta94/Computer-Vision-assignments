require 'nn'
require 'image'
require 'torch'
require 'env'
require 'trepl'


local cmd = torch.CmdLine()
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-batchsize', 100, 'batchsize')
cmd:option('-mnist', false, 'use mnist')
cmd:option('-cifar', false, 'use cifar')
cmd:option('-epochs', 10 , 'epochs')
local config = cmd:parse(arg)

local tnt   = require 'torchnet'
local dbg   = require 'debugger'
-- to set breakpoint put just put: dbg() at desired line

local base_data_path = '/Users/abhinav/Documents/Acads/CV/CompVision/assignment-2/data/'

-- Dataprep for MNIST

if config.mnist == true then
    if not paths.filep(base_data_path .. 'train_small_28x28.t7') then
        local train = torch.load(base_data_path .. 'train_28x28.t7', 'ascii')
        local train_small = {}
        train_small.data   = train.data[{{1, 50000}, {}, {}, {}}]
        train_small.labels = train.labels[{{1, 50000}}]
        torch.save(base_data_path .. 'train_small_28x28.t7', train_small, 'ascii')
    end

    if not paths.filep(base_data_path .. 'valid.t7') then
        local train = torch.load(base_data_path .. 'train_28x28.t7', 'ascii')
        local valid = {}
        valid.data   = train.data[{{50001, 60000}, {}, {}, {}}]
        valid.labels = train.labels[{{50001, 60000}}]
        torch.save(base_data_path .. 'valid_28x28.t7', valid, 'ascii')
    end
end

------------------------------------------------------------------------
-- Build the dataloader

-- getDatasets returns a dataset that performs some minor transformation on
-- the input and the target (TransformDataset), shuffles the order of the
-- samples without replacement (ShuffleDataset) and merges them into
-- batches (BatchDataset).
local function getMnistIterator(datasets)
    local listdatasets = {}
    for _, dataset in pairs(datasets) do
     --   local list = torch.range(1, dataset.data:size(1)):totable()
       local list = torch.range(1, 50):totable()
        table.insert(listdatasets,
                    tnt.ListDataset{
                        list = list,
                        load = function(idx)
                            return {
                                input  = dataset.data[idx],
                                target = dataset.labels[idx]
                            } -- sample contains input and target
                        end
                    })
    end
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = config.batchsize,
            dataset = tnt.ShuffleDataset{
               dataset = tnt.TransformDataset{
                    transform = function(x)
		                    return {
	                            input  = x.input:view(-1):double(),
		                        target = torch.LongTensor{x.target + 1}
                            }
                    end,
                    dataset = tnt.ConcatDataset{
                        datasets = listdatasets
                    }
                },
            }
        }
    }
end

local function getCifarIterator(datasets)
    local listdatasets = {}
    for _, dataset in pairs(datasets) do
        local list = torch.range(1, dataset.data:size(1)):totable()
        table.insert(listdatasets,
                    tnt.ListDataset{
                        list = list,
                        load = function(idx)
                            return {
                                input  = dataset.data[{{}, idx}],
                                target = dataset.labels[{{}, idx}]
                            } -- sample contains input and target
                        end
                    })
    end
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = config.batchsize,
            dataset = tnt.ShuffleDataset{
               dataset = tnt.TransformDataset{
                    transform = function(x)
		       return {
			  input  = x.input:double():reshape(3,32,32),
			  target = x.target:long():add(1),
		       }
                    end,
                    dataset = tnt.ConcatDataset{
                        datasets = listdatasets
                    }
                },
            }
        }
    }
end

------------------------------------------------------------------------
-- Make the model and the criterion

local nout = 10 --same for both CIFAR and MNIST
local nin
if config.mnist == true then nin = 784 end
if config.cifar == true then nin = 3072 end

-- local cnn = nn.Linear(nin, nout)

local cnn = nn.Sequential()
cnn:add(nn.SpatialConvolution(3,16,5,5))
cnn:add(nn.Tanh())
cnn:add(nn.SpatialMaxPooling(2,2))
cnn:add(nn.SpatialConvolution(16,128,5,5))
cnn:add(nn.Tanh())
cnn:add(nn.SpatialMaxPooling(2,2))
cnn:add(nn.View(128*5*5))
cnn:add(nn.Linear(128*5*5, 64))
cnn:add(nn.Tanh())
cnn:add(nn.Linear(64, nout))

local criterion = nn.CrossEntropyCriterion()

------------------------------------------------------------------------
-- Prepare torchnet environment for training and testing

local trainiterator
local validiterator
local testiterator
if config.mnist == true then
    local datasets
    datasets = {torch.load(base_data_path .. 'train_small_28x28.t7', 'ascii')}
    trainiterator = getMnistIterator(datasets)
    datasets = {torch.load(base_data_path .. 'valid_28x28.t7', 'ascii')}
    validiterator = getMnistIterator(datasets)
    datasets = {torch.load(base_data_path .. 'test_28x28.t7', 'ascii')}
    testiterator  = getMnistIterator(datasets)
end
if config.cifar == true then
    local datasets
    datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_1.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_2.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_3.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_4.t7', 'ascii')}
    trainiterator = getCifarIterator(datasets)
    datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_5.t7', 'ascii')}
    validiterator = getCifarIterator(datasets)
    datasets = {torch.load(base_data_path .. 'cifar-10-torch/test_batch.t7', 'ascii')}
    testiterator  = getCifarIterator(datasets)
end

local lr = config.lr
local epochs = config.epochs

print("Started training!\n")

for epoch = 1, epochs do
    local timer = torch.Timer()
    local loss = 0
    local errors = 0
    local count = 0
    for d in trainiterator() do
        cnn:forward(d.input)
        criterion:forward(cnn.output, d.target)
        cnn:zeroGradParameters()
        criterion:backward(cnn.output, d.target)
        cnn:backward(d.input, criterion.gradInput)
        cnn:updateParameters(lr)

        loss = loss + criterion.output --criterion already averages over minibatch
        count = count + 1
        local _, pred = cnn.output:max(2)
        errors = errors + (pred:size(1) - pred:eq(d.target):sum())
    end
    loss = loss / count


    local validloss = 0
    local validerrors = 0
    count = 0
    for d in validiterator() do
        cnn:forward(d.input)
        criterion:forward(cnn.output, d.target)

        validloss = validloss + criterion.output --criterion already averages over minibatch
        count = count + 1
        local _, pred = cnn.output:max(2)
        validerrors = validerrors + (pred:size(1) - pred:eq(d.target):sum())
    end
    validloss = validloss / count

    print(string.format(
    'train | epoch = %d | lr = %1.4f | loss: %2.4f | error: %2.4f - valid | validloss: %2.4f | validerror: %2.4f | s/iter: %2.4f\n',
    epoch, lr, loss, errors, validloss, validerrors, timer:time().real
    ))
end

local testerrors = 0
for d in testiterator() do
    cnn:forward(d.input)
    criterion:forward(cnn.output, d.target)
    local _, pred = cnn.output:max(2)
    testerrors = testerrors + (pred:size(1) - pred:eq(d.target):sum())
end

print(string.format('| test | error: %2.4f\n', testerrors))

--  code for plotting weights
print(cnn)
print(#cnn:getParameters())

torch.save("cnn.t7", cnn:clearState())

weights = cnn:get(1).weight

image.savePNG("weight_cnn.png", image.toDisplayTensor{input = weights, padding = 2})