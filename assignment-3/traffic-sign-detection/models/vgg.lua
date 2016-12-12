require 'nn'

local vgg = nn.Sequential()

vgg:add(nn.SpatialConvolution(3, 64, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(64,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(64, 64, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(64,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.SpatialMaxPooling(2,2,2,2))

vgg:add(nn.SpatialConvolution(64, 128, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(128,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(128, 128, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(128,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.SpatialMaxPooling(2,2,2,2))

vgg:add(nn.SpatialConvolution(128, 256, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(256,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(256, 256, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(256,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(256, 256, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(256,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(256, 256, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(256,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.SpatialMaxPooling(2,2,2,2))

vgg:add(nn.SpatialConvolution(256, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.SpatialMaxPooling(2,2,2,2))

vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.SpatialMaxPooling(2,2,2,2))

vgg:add(nn.View(512))

local classifier = nn.Sequential()

classifier:add(nn.View(512))
classifier:add(nn.Linear(512, 512))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(512, 1e-3))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 512))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(512, 1e-3))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 43))

vgg:add(classifier)

return vgg
