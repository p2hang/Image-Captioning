require 'torch'
require 'optim'
require 'os'
require 'xlua'
ld = require "util/load_data"
-- require 'cunn'
-- require 'cudnn'
require 'loadcaffe'

local tnt = require 'torchnet'
-- local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local train_set_size = 4141113
local val_set_size = 201654


ImgCap = {}


torch.setdefaulttensortype('torch.DoubleTensor')

torch.manualSeed(opt.manualSeed)
-- cutorch.manualSeedAll(opt.manualSeed)
require('ImgCapModel/' .. opt.model)

-- data transformation




--function getTestSample(dataset, idx)
--end

-- get iterator
function getIterator(data_type)
    assert(data_type == "train" or data_type == "val")
    return tnt.ParallelDatasetIterator{
        nthread = 1,
        init = function() 
            require 'torchnet' 
            ld = require "util/load_data"
            
            end,
        closure = function()
            local dataset = ld:loadData(data_type)
            require 'image'
            local WIDTH, HEIGHT = 224, 224
            function resize(img)
                return image.scale(img, WIDTH,HEIGHT)
            end

            function padInput(ip)
                local input = torch.IntTensor(32):fill(1)
                if ip:size()[1] < 32 then
                    len = ip:size()[1]
                else
                    len = 31
                end
                input[1] = 2
                for i = 1, len do 
                    input[i + 1] = input[i]
                end
                return input
            end


            function transformInput(inp)
                local sample = {}
                local f = tnt.transform.compose{
                    [1] = resize
                }
                local fc = tnt.transform.compose{
                    [1] = padInput
                }
                sample.image = f(inp.image)
                sample.caption = fc(inp.caption)
                return sample 
            end

            function padTarget(tg)
                local target = torch.IntTensor(32):fill(1)
                if tg:size()[1] < 32 then
                    len = tg:size()[1]
                else
                    len = 31
                end

                for i = 1, len do 
                    target[i] = tg[i]
                end
                target[len + 1] = 3
                return target 
            end

            function transformTarget(tg)
                local f = tnt.transform.compose{
                    [1] = padTarget
                }
                return f(tg)
            end

            return tnt.BatchDataset{
                batchsize = 2,
                dataset = tnt.ListDataset{
                    list = torch.range(1, #dataset):long(),
                    load = function(idx)
                        return{
                            input = transformInput({['image']= ld:loadImage(data_type, dataset[idx].image_id),
                                     ['caption'] = dataset[idx].caption}),
                            target = transformTarget(dataset[idx].caption)
                        }
                    end,
                }
            }
        end,
    }
end




local config = {} -- config for model, default as vgg
config.embeddingDim = opt.embeddingDim
config.imageOutputLayer = opt.imageLayer
config.imageModelPrototxt = 'models/caffe/' .. opt.protobuf or 'VGG_ILSVRC_19_layers_deploy.prototxt'
config.imageModelBinary = 'models/caffe/' .. opt.caffemodel or 'VGG_ILSVRC_19_layers.caffemodel'
config.num_words = opt.numWords

local model = ImgCap.vggLSTM(config)





local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
local clerr = tnt.ClassErrorMeter{topk = {3}}
local timer = tnt.TimeMeter()
local batch = 1 

if opt.cuda then 
    print("Using CUDA")
    require 'cunn'
    require 'cudnn'
    require 'cutorch'
    cutorch.setDevice(opt.gpuid)
    model:cuda()
    criterion:cuda()
    local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
        engine.hooks.onSample = function(state)
        igpu:resize(state.sample.input:size()):copy(state.sample.input)
        state.sample.input = igpu
        if state.sample.target then
            tgpu:resize(state.sample.target:size()):copy(state.sample.target)
            state.sample.target = tgpu
        end
    end
end


engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
        num_iters = train_set_size / opt.batchsize
    else
        mode = 'Val'
        num_iters = val_set_size / opt.batchsize
    end
end

engine.hooks.onForward = function(state)
    -- state.sample.target = state.sample.target[1]
end

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)



    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, num_iters, meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, num_iters)
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end


engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
end

local lr = opt.LR 
local epoch = 1
while epoch <= opt.nEpochs do 
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator('train'),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = lr,
            momentum = opt.momentum
        }
    }


    epoch = epoch + 1
end



