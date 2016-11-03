require 'torch'
require 'image'

mnist_images = torch.load("data/train_32x32.t7", "ascii")
cifar_images = torch.load("data//cifar-10-torch/data_batch_1.t7", "ascii")
mnist_data = mnist_images.data
c = cifar_images.data:permute(2,1)
cifar_data = c:reshape(c:size(1),3,32,32)

files = {}
for i=1,100 do
	table.insert(files, cifar_data[i])
end

image.savePNG("cifar100.png", image.toDisplayTensor{input=files, padding = 2})

files = {}
for i=1,100 do
	table.insert(files, mnist_data[i])
end

image.savePNG("mnist100.png", image.toDisplayTensor{input=files, padding = 2})