require 'torch'
require 'optim'
require 'os'
require 'xlua'
require 'math'
ld = require "util/load_data"
-- require 'cunn'
-- require 'cudnn'
require 'loadcaffe'

local tnt = require 'torchnet'
-- local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local train_set_size = 414113
local val_set_size = 201654
local min_error = 100 -- err of the best model
local current_error = 100 -- err of current bat

ImgCap = {}


torch.setdefaulttensortype('torch.DoubleTensor')

torch.manualSeed(opt.manualSeed)
-- cutorch.manualSeedAll(opt.manualSeed)
require('ImgCapModel/' .. opt.model)


--function getTestSample(dataset, idx)
--end

-- get iterator
function getIterator(data_type)
    assert(data_type == "train" or data_type == "val")
    return tnt.ParallelDatasetIterator{
        nthread = opt.nThreads,
        init = function() 
            require 'torchnet' 
            ld = require "util/load_data"
            end,
        closure = function()
            local dataset = ld:loadData(data_type)
            local trans = require 'util/transform'
            tnt.BatchDataset.get = require 'util/get_in_batchdataset'

            return tnt.BatchDataset{
                batchsize = opt.batchsize,
                dataset = tnt.ListDataset{
                    list = torch.range(1, #dataset):long(),
                    load = function(idx)
                        return {
                            image = trans.onInputImage(ld:loadImage(data_type, dataset[idx].image_id)),
                            text = trans.onInputText(dataset[idx].caption),
                            target = trans.onTarget(dataset[idx].caption)
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
config.batchsize = opt.batchsize

local model = ImgCap.vggLSTM(config)


-- Init model from previous trained result.
if opt.useWeights then
    print("Use pretrained weights for the vggLSTM.")
    model = torch.load(opt.weights .. opt.model .. '_weights.t7')
end




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
        num_iters = math.floor(train_set_size / opt.batchsize)
    else
        mode = 'Val'
        num_iters = math.floor(val_set_size / opt.batchsize)
    end
end

engine.hooks.onForward = function(state)
    print(state.sample.target)
    -- local batch_size = #state.sample.target

    -- local caption_size = state.sample.target[1]:size()
    -- local caption_tensor = torch.DoubleTensor(batch_size, caption_size[1])

    -- for i = 1, batch_size do 
    --     caption_tensor[i]:copy(state.sample.target[i])
    -- end

    -- state.sample.target = caption_tensor
    -- print()
    -- print("done onforward")
end

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    --clerr:add(state.network.output, state.sample.target)
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

    -- the error on end of the training
    current_error = clerr:value { k = 1 }
end

local lr = opt.LR 
local epoch = 1

print("Start Training!")
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

    engine:test {
        network = model,
        criterion = criterion,
        iterator = getIterator('val')
    }

    -- save the model if it is the best so far
    if current_error <= min_error then
        print("update model")
        min_error = current_error
        torch.save(opt.weights .. opt.model .. '_weights.t7', model:clearState())
    end

    print('Done with Epoch ' .. tostring(epoch))
    epoch = epoch + 1
end


print("The End!")
