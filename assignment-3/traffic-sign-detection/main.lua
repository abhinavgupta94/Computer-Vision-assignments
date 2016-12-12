require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = opt.psize, opt.psize
local channels = {'y','u','v'}
local neighborhood = image.gaussian(5)
local normalization = nn.SpatialContrastiveNormalization(1, neighborhood)

local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

torch.setdefaulttensortype('torch.DoubleTensor')

torch.setnumthreads(opt.nThreads)
if opt.cuda == 'true' then
    require 'cunn'
    require 'cudnn'
    cudnn.benchmark = true
    cudnn.fastest = true
    cutorch.manualSeedAll(opt.manualSeed)
else
    torch.manualSeed(opt.manualSeed)
end

function crop(img,x1,y1,x2,y2)
    return image.crop(img, x1, y1, x2, y2)
end

function resize(img)
    return image.scale(img, WIDTH, HEIGHT)
end

function norm(img)
    for c in ipairs(channels) do
        img[{ {c},{},{} }]:add(-img[{ {c},{},{} }]:mean())
        img[{ {c},{},{} }]:div(img[{ {c},{},{} }]:std())
    end
    return img
end

local function localnorm(img)
    for c in ipairs(channels) do
        img[{ {c},{},{} }] = normalization:forward(img[{ {c},{},{} }])
    end
    return img
end

function translatejitter(img)
    rand_position_x = (torch.randn(1)*2)[1]
    rand_position_y = (torch.randn(1)*2)[1]
    return image.translate(img, rand_position_x, rand_position_y)
end

function rotatejitter(img)
    rand_angle = (torch.randn(1)*15*3.14/180)[1]
    return image.rotate(img, rand_angle)
end

local function transformJitter(img)
    f = tnt.transform.compose{
        [1] = localnorm,
        [2] = rotatejitter,
        [3] = translatejitter,
        [4] = resize,
        [5] = norm
    }
    return f(img)
end

local trainData = torch.load(DATA_PATH..'train.t7')
local testData = torch.load(DATA_PATH..'test.t7')
local trainDataImages = torch.Tensor(trainData:size(1), 3, WIDTH, HEIGHT)
local testDataImages = torch.Tensor(testData:size(1), 3, WIDTH, HEIGHT)
local trainLabel = torch.Tensor(trainData:size(1),1)
local testLabel = torch.Tensor(testData:size(1),1)
-- for upsampling---------------------------------------------------------
-- local totalData = {}
-- local valData = {}
-- for i=1,43 do
--     totalData[i] = {}
--     valData[i] = {}
-- end

-- open preloaded train images (shoud be true for the first time)
if opt.reload == true then

    local mean = {}
    local std = {}

    print("Starting preprocessing")
    for j = 1,math.floor(trainData:size(1)) do
        r = trainData[j]
        classId, track, file = r[9], r[1], r[2]
        x1, y1, x2, y2 = r[5], r[6], r[7], r[8]
        file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
        preim = image.load(DATA_PATH .. '/train_images/'..file)
        -- crop ROI's flag-------------------
        if opt.crop == true then
            preim = crop(preim, x1, y1, x2, y2)
        end
        img = resize(preim, WIDTH, HEIGHT)
        trainDataImages[j] = img
        trainLabel[j] = torch.LongTensor{r[9] + 1}
        -- table.insert(totalData[r[9]+1],j) -- for upsampling
    end
    print("Train Data loaded")

    for j = 1,testData:size(1) do
        r = testData[j]
        x1, y1, x2, y2 = r[4], r[5], r[6], r[7]
        file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
        preim = image.load(file)
        -- crop ROI's flag-------------------
        if opt.crop == true then
            preim = crop(preim, x1, y1, x2, y2)
        end
        img = resize(preim, WIDTH, HEIGHT)
        testDataImages[j] = img
        testLabel[j] = torch.LongTensor{r[1]}
    end
    print("Test Data loaded")

    -- Upsampling starts---------------------------------------------------------
    -- extract 10% from the training data
    -- for i = 1,43 do
    --     size = #totalData[i]
    --     for j = 1, math.floor(size/opt.val) do
    --         table.insert(valData[i], totalData[i][j])
    --         table.remove(totalData[i], j)
    --     end
    -- end

    -- get maximum count from all classes
    -- classCounts = torch.Tensor(43)
    -- for i = 1,43 do
    --     classCounts[i] = #totalData[i]
    -- end
    -- maxCount = torch.max(classCounts)
    -- -- print(maxCount)

    -- create new training dataset with equal instances of all classes
    -- for i = 1,43 do
    --     size = #totalData[i]
    --     actualSize = size
    --     while size < maxCount do
    --         val = torch.random(1,actualSize)
    --         table.insert(totalData[i],val)
    --         size = size + 1
    --     end
    -- end

    -- local finalData = torch.Tensor()
    -- for i = 1,43 do
    --     if i == 1 then 
    --         finalData = torch.Tensor(totalData[i])
    --     else
    --         finalData = torch.cat(finalData,torch.Tensor(totalData[i]))
    --     end
    -- end

    -- local validData = torch.Tensor()
    -- for i = 1,43 do
    --     if i == 1 then 
    --         validData = torch.Tensor(valData[i])
    --     else
    --         validData = torch.cat(validData,torch.Tensor(valData[i]))
    --     end
    -- end

    -- print("Upsampling done")
    -- print("Size of Training set " .. finalData:size(1))
    -- print("Size of Validation set " .. validData:size(1))
    -- print("Size of Test set " .. testData:size(1))

    -- convert to YUV scale-----------------------------------------------------
    for i = 1,trainData:size(1) do
        trainDataImages[i] = image.rgb2yuv(trainDataImages[i])
    end

    for i = 1,testData:size(1) do
        testDataImages[i] = image.rgb2yuv(testDataImages[i])
    end
    -- Global normalization-----------------------------------------------------
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
    print("Global Normalization done")
    if opt.crop == true then
        suff = "_c"
    else
        suff = tostring(opt.psize)
    end
    torch.save(DATA_PATH.."trainImages".. suff .. ".t7", trainDataImages)
    torch.save(DATA_PATH.."testImages" .. suff .. ".t7", testDataImages)
    torch.save(DATA_PATH.."trainLabels.t7", trainLabel)
    torch.save(DATA_PATH.."testLabels.t7", testLabel)

else
    if opt.crop == true then
        suff = "_c"
    else
        suff = tostring(opt.psize)
    end
    trainDataImages = torch.load(DATA_PATH.."trainImages" .. suff .. ".t7")
    testDataImages = torch.load(DATA_PATH.."testImages" .. suff .. ".t7")
    trainLabel = torch.load(DATA_PATH.."trainLabels.t7")
    testLabel = torch.load(DATA_PATH.."testLabels.t7")
    print("Data loaded")
end

-- jitter and original dataset if upsampling is enabled
-- local jittdataset = tnt.ListDataset{
--     list = torch.range(1, finalData:size(1)*opt.jitters):long(),
--     load = function(idx)
--         return {
--             input = transformJitter(trainDataImages[finalData[math.ceil(idx/opt.jitters)]]),
--             target = trainLabel[finalData[math.ceil(idx/opt.jitters)]]
--         }
--     end
-- }

-- local origdataset = tnt.ListDataset{
--     list = torch.range(1, finalData:size(1)):long(),
--     load = function(idx)
--         return {
--             input = localnorm(trainDataImages[finalData[idx]]),
--             target = trainLabel[finalData[idx]]
--         }
--     end
-- }

local jittdataset = tnt.ListDataset{
    list = torch.range(1, trainDataImages:size(1)*opt.jitters):long(),
    load = function(idx)
        return {
            input = transformJitter(trainDataImages[math.ceil(idx/opt.jitters)]),
            target = trainLabel[math.ceil(idx/opt.jitters)]
        }
    end
}

local origdataset = tnt.ListDataset{
    list = torch.range(1, trainDataImages:size(1)):long(),
    load = function(idx)
        return {
            input = localnorm(trainDataImages[idx]),
            target = trainLabel[idx]
        }
    end
}
-- concatenate jittered and original dataset
local trainDataset = tnt.ShuffleDataset{
    dataset = tnt.ConcatDataset{
        datasets = {jittdataset, origdataset}
    }
}

-- validation data extracted before upsampling
-- local valDataset = tnt.ListDataset{
--     list = torch.range(1, validData:size(1)):long(),
--     load = function(idx)
--         return {
--             input = localnorm(trainDataImages[validData[idx]]),
--             target = trainLabel[validData[idx]]
--         }
--     end
-- }

local testDataset = tnt.ListDataset{
    list = torch.range(1, testDataImages:size(1)):long(),
    load = function(idx)
        return {
            input = localnorm(testDataImages[idx]),
            sampleId = testLabel[idx]
        }
    end
}

function getTestIterator()

    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = testDataset
        }
    }
end
-- Parallel Iterator for training
function getParallelIterator(dataset)
        
    return tnt.ParallelDatasetIterator{
        nthread = opt.nThreads,
        init    = function() local tnt = require 'torchnet' end,
        closure = function()

        return tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
        }
        end,
    } 
end

local model = require("models/".. opt.model)
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

-- print(model)
if opt.cuda == true then
    cudnn.convert(model, cudnn)
    -- model = model:cuda()
    criterion = criterion:cuda()
    local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
    engine.hooks.onSample = function(state)
        igpu:resize(state.sample.input:size()):copy(state.sample.input)
        tgpu:resize(state.sample.target:size()):copy(state.sample.target)
        state.sample.input  = igpu
        state.sample.target = tgpu
    end
end

engine.hooks.onStart = function(state)
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

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == 'true' then
        print(string.format("%s Batch: %d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, meter:value(), clerr:value{k = 1}))
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
end

local epoch = 1

while epoch <= opt.nEpochs do
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getParallelIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            momentum = opt.momentum
        }
    }
    -- engine:test{
    --     network = model,
    --     criterion = criterion,
    --     iterator = getParallelIterator(valDataset)
    -- }
    -- if epoch <=50 then
    --     if epoch < 25 and epoch == 20 then
    --         learningRate = learningRate / 2
    --     elseif epoch %10 == 0 then
    --            learningRate = learningRate / 2 
    --     end
    -- end
    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
    torch.save(opt.sub .. ".t7", model)

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

if opt.cuda == 'true' then
    local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
    engine.hooks.onSample = function(state)
        igpu:resize(state.sample.input:size()):copy(state.sample.input)
        tgpu:resize(state.sample.sampleId:size()):copy(state.sample.sampleId)
        state.sample.input  = igpu
        state.sample.sampleId = tgpu
    end
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network = model,
    iterator = getTestIterator()
}

torch.save(opt.sub .. ".t7", model:clearState())

print("The End!")
