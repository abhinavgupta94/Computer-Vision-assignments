require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
require 'cunn'
require 'cudnn' -- faster convolutions

--[[
--  Hint:  Plot as much as you can.  
--  Look into torch wiki for packages that can help you plot.
--]]

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 32, 32
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

-- torch.setdefaulttensortype('torch.DoubleTensor')

-- torch.setnumthreads(1)
-- torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)


function crop(img,x1,y1,x2,y2)
    return image.crop(img, x1, y1, x2, y2)
end

function resize(img)
    return image.scale(img, WIDTH, HEIGHT)
end

function globalnorm(img)
    img = img - torch.mean(img)
    -- img[{ {1},{},{} }]:add(-img[{ {1},{},{} }]:mean())
    -- img[{ {1},{},{} }]:div(img[{ {1},{},{} }]:std())
    -- img[{ {2},{},{} }]:add(-img[{ {2},{},{} }]:mean())
    -- img[{ {2},{},{} }]:div(img[{ {2},{},{} }]:std())
    -- img[{ {3},{},{} }]:add(-img[{ {3},{},{} }]:mean())
    -- img[{ {3},{},{} }]:div(img[{ {3},{},{} }]:std())
    return img
end

function localnorm(img)
    normalization = nn.SpatialContrastiveNormalization(3, image.gaussian(5))
    return normalization:forward(img)
end

--[[
-- Hint:  Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing 
-- information by resizing bigger images to a smaller size?
--]]
function transformInput(img)
    f = tnt.transform.compose{
        [1] = resize,
        [2] = globalnorm
        -- [3] = localnorm
    }
    return f(img)
end

function translatejitter(img)
    local rand_position_x = (torch.randn(1)*2)[1]
    local rand_position_y = (torch.randn(1)*2)[1]
    return image.translate(img, rand_position_x, rand_position_y)   
end

function rotatejitter(img)
    local rand_angle = (torch.randn(1)*15*3.14/180)[1]
    return image.rotate(img, rand_angle)
end

function transformJitter(img)
    f = tnt.transform.compose{
        [1] = rotatejitter,
        [2] = translatejitter,
        [3] = resize,
        [4] = globalnorm
    }
    return f(img)
end

function normalizeJitter(img)
    f = tnt.transform.compose{
        [1] = resize,
        [2] = globalnorm
        -- [3] = localnorm
    }
    return f(img)
end

function getJitteredSample(dataset, idx)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    x1, y1, x2, y2 = r[5], r[6], r[7], r[8]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    transformed = transformJitter(image.load(DATA_PATH .. '/train_images/'..file))
    -- newval = torch.randn(1)*0.2 + 0.9
    newval = 1
    cropped = crop(transformed, newval*x1, newval*y1, newval*x2, newval*y2)
    return normalizeJitter(cropped)
end

function getTrainSample(dataset, idx)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    x1, y1, x2, y2 = r[5], r[6], r[7], r[8]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    cropped = crop(image.load(DATA_PATH .. '/train_images/'..file), x1, y1, x2, y2)
    return transformInput(cropped)
end

function getTrainLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file))
end

function getIterator(dataset)
    --[[
    -- Hint:  Use ParallelIterator for using multiple CPU converges
    
    return tnt.ParallelDatasetIterator{
        nthread = 1,
        init    = function() require 'torchnet' end,   
        closure = function()


        local trainData = torch.load(DATA_PATH..'train.t7')
        local image = require 'image'
        local WIDTH, HEIGHT = 32, 32

        dataset = tnt.SplitDataset{
            partitions = {train=0.9, val=0.1},
            initialpartition = 'train',
            
            dataset = tnt.ShuffleDataset{
                dataset = tnt.ListDataset{
                    list = torch.range(1, trainData:size(1)):long(),
                    load = function(idx)
                        return {
                            input = image.scale(image.load(
                                DATA_PATH .. '/train_images/'..
                                string.format("%05d/%05d_%05d.ppm", trainData[idx][9], trainData[idx][1], trainData[idx][2])), WIDTH, HEIGHT),
                            target = torch.LongTensor{dataset[idx][9] + 1}
                        }
                    end
                }
            }
        }

        return tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
        }
        end,
    }
    --]]
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
        }
    }
end

local trainData = torch.load(DATA_PATH..'train.t7')
local testData = torch.load(DATA_PATH..'test.t7')

channels = {'y','u','v'}

mean = {}
std = {}
trainDataImages = torch.Tensor(trainData:size(1), 3, WIDTH, HEIGHT)
testDataImages = torch.Tensor(testData:size(1), 3, WIDTH, HEIGHT)
trainLabel = {}
testLabel = {}

for j = 1,trainData:size(1) do
    r = trainData[j]
    classId, track, file = r[9], r[1], r[2]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    img = resize(image.load(DATA_PATH .. '/train_images/'..file), WIDTH, HEIGHT)
    trainDataImages[j] = img
    trainLabel[j] = torch.LongTensor{r[9] + 1}
end

for j = 1,testData:size(1) do
    r = testData[j]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    img = resize(image.load(file), WIDTH, HEIGHT)
    testDataImages[j] = img
    testLabel[j] = torch.LongTensor{r[1]}
end

for i = 1,trainData:size(1) do
   trainDataImages[i] = image.rgb2yuv(trainDataImages[i])
end

for i = 1,testData:size(1) do
   testDataImages[i] = image.rgb2yuv(testDataImages[i])
end

for i,channel in ipairs(channels) do
   mean[i] = trainDataImages[{ {},i,{},{} }]:mean()
   std[i] = trainDataImages[{ {},i,{},{} }]:std()
   trainDataImages[{ {},i,{},{} }]:add(-mean[i])
   trainDataImages[{ {},i,{},{} }]:div(std[i])
end

for i,channel in ipairs(channels) do
   testDataImages[{ {},i,{},{} }]:add(-mean[i])
   testDataImages[{ {},i,{},{} }]:div(std[i])
end

neighborhood = image.gaussian(5)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood)

for c in ipairs(channels) do
   for i = 1,trainData:size(1) do
      trainDataImages[{ i,{c},{},{} }] = normalization:forward(trainDataImages[{ i,{c},{},{} }])
   end
   for i = 1,testData:size(1) do
      testDataImages[{ i,{c},{},{} }] = normalization:forward(testDataImages[{ i,{c},{},{} }])
   end
end

-- for i = 1,trainData:size(1) do
--    trainDataImages[i] = image.yuv2rgb(trainDataImages[i])
-- end

-- for i = 1,testData:size(1) do
--    testDataImages[i] = image.yuv2rgb(testDataImages[i])
-- end

jittdataset = tnt.ListDataset{
    list = torch.range(1, trainData:size(1)*5):long(),
    load = function(idx)
        return {
            input = transformJitter(trainDataImages[math.ceil(idx/5)]),
            target = trainLabel[math.ceil(idx/5)]
        }
    end
}

origdataset = tnt.ListDataset{
    list = torch.range(1, trainData:size(1)):long(),
    load = function(idx)
        return {
            input = trainDataImages[idx],
            target = trainLabel[idx]
        }
    end
}

trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ConcatDataset{
            datasets = {jittdataset, origdataset}
        }
    }
}

testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = testDataImages[idx],
            sampleId = testLabel[idx]
        }
    end
}

--[[
-- Hint:  Use :cuda to convert your model to use GPUs
--]]
local model = require("models/".. opt.model):cuda()
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion():cuda()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

-- print(model)

engine.hooks.onStartEpoch = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end

--[[
-- Hint:  Use onSample function to convert to 
--        cuda tensor for using GPU
--]]
-- engine.hooks.onSample = function(state)
-- end

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
engine.hooks.onSample = function(state)
    igpu:resize(state.sample.input:size()):copy(state.sample.input)
    tgpu:resize(state.sample.target:size()):copy(state.sample.target)
    state.sample.input  = igpu
    state.sample.target = tgpu
end  -- alternatively, this logic can be implemented via a TransformDataset 

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
end

local epoch = 1

while epoch <= opt.nEpochs do
    trainDataset:select('train')
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            momentum = opt.momentum
        }
    }

    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }
    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

local submission = assert(io.open(opt.logDir .. "/" .. opt.sub .. ".csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
    local fileNames  = state.sample.sampleId
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
engine.hooks.onSample = function(state)
    igpu:resize(state.sample.input:size()):copy(state.sample.input)
    tgpu:resize(state.sample.sampleId:size()):copy(state.sample.sampleId)
    state.sample.input  = igpu
    state.sample.sampleId = tgpu
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

print("The End!")
