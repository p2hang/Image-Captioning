require 'torch'
require 'loadcaffe'
require 'csvigo'
require 'cunn'
require 'cudnn'
require 'image'

dataset = 'train'

local result = {}

local trans = require 'util.transform'
local tnt = require 'torchnet'
local tntTrans = require 'torchnet.transform'
local makebatch =  tntTrans.makebatch{merge=merge}

local imageModelPrototxt = 'models/caffe/VGG_ILSVRC_19_layers_deploy.prototxt'
local imageModelBinary = 'models/caffe/VGG_ILSVRC_19_layers.caffemodel'

local model = loadcaffe.load(imageModelPrototxt, imageModelBinary)
for i = #model.modules, 40 + 1, -1 do
    model:remove(i)
end

model = model:cuda()
cudnn.convert(model, cudnn)
list = csvigo.load{ path='data/'..dataset..'list1.txt' }.data



batch_size = 40
--------------------
-- extract
--------------------
for i = 0, #list / batch_size + 1  do

    curBatch = {}
    for j = 1, batch_size do
        if i * batch_size + j > #list then break end

        img = image.load('data/'..dataset..'2014/'..list[i*batch_size + j])
        rescaled_img =  trans.onInputImage(img)
        sample = {['img']=rescaled_img}
        table.insert(curBatch, sample)
    end
    curBatch = makebatch(curBatch)
    output = model:forward(curBatch.img:cuda()):float()

    for j = 1, output:size(1) do
      image_id = string.split(string.split(list[i*batch_size + j], '_')[3], "%.")[1]
      result[image_id] = output[j]
    end
    
    collectgarbage()
    print(i)

    if (i + 1) * batch_size >= #list then break end
end



torch.save(dataset.."_features1.t7", result)


