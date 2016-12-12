local modelType = 'E' -- on a titan black, B/D/E run out of memory even for batch-size 32
local po = 1
local utils = paths.dofile('utils.lua')
-- Create tables describing VGG configurations A, B, D, E
local cfg = {}
if modelType == 'A' then
   cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512}
elseif modelType == 'B' then
   cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512}
elseif modelType == 'D' then
   cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512}
elseif modelType == 'E' then
   cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512}
else
   error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
end

local features = nn.Sequential()
do
   local iChannels = 3
   for k,v in ipairs(cfg) do
      if v == 'M' then
         features:add(nn.SpatialMaxPooling(2,2,2,2))
      else
         local oChannels = v
         local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1):noBias()
         local batchnorm = nn.SpatialBatchNormalization(oChannels,1e-3)
         features:add(conv3)
         features:add(batchnorm)
         features:add(nn.ReLU(true))
         features:add(nn.Dropout(0.4))
         iChannels = oChannels
      end
   end
end
features:add(nn.SpatialAveragePooling(2,2,2,2))
features:cuda()
-- features = makeDataParallel(features, nGPU) -- defined in util.lua

local classifier = nn.Sequential()
classifier:add(nn.View(512*po*po))
classifier:add(nn.Linear(512*po*po, 512*po))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(512*po, 1e-3))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512*po, 512))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(512, 1e-3))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 43))
-- classifier:add(nn.LogSoftMax())
classifier:cuda()

local model = nn.Sequential()
model:add(features):add(classifier)

utils.FCinit(model)
utils.MSRinit(model)

return model
