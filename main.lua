require 'torch'
require 'optim'
require 'os'
require 'xlua'
require 'math'
ld = require "util/load_data"
-- require 'cunn'
-- require 'cudnn'
-- require 'loadcaffe'

local tnt = require 'torchnet'
local optParser = require 'opts'
local opt = optParser.parse(arg)
-- torch.setdefaulttensortype('torch.FloatTensor')

local train_set_size = 414113
local val_set_size = 201654
local min_loss = 100 -- err of the best model
local current_loss = 100 -- err of current bat

ImgCap = {}

logger = optim.Logger('logs/loss.log')
logger:setNames{'Training Loss', 'Validation Loss'}
local trainLoss
local valLoss

torch.setdefaulttensortype('torch.DoubleTensor')

torch.manualSeed(opt.manualSeed)
require('ImgCapModel/' .. opt.model)

-----------------------------------------------------------------------
---- Get the dataset interator and transformation
-----------------------------------------------------------------------

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
                        -- print(dataset[idx].image_id)
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

-----------------------------------------------------------------------
---- Config or load the model.
-----------------------------------------------------------------------


local config = {} -- config for model, default as vgg
config.embeddingDim = opt.embeddingDim
config.imageOutputLayer = opt.imageLayer
config.imageModelPrototxt = 'models/caffe/' .. opt.protobuf or 'VGG_ILSVRC_19_layers_deploy.prototxt'
config.imageModelBinary = 'models/caffe/' .. opt.caffemodel or 'VGG_ILSVRC_19_layers.caffemodel'
config.num_words = opt.numWords
config.batchsize = opt.batchsize


local model

-- Init model from previous trained result.
if opt.cuda and  opt.useWeights then
    require 'cunn'
    require 'cudnn'
    require 'cutorch'
    print("Use pretrained weights for the vggLSTM.")
    model = torch.load(opt.weightsDir .. opt.model .. '_weights.t7')
    -- model:float()
    -- cudnn.convert(model, cudnn)
else 
    model = ImgCap.vggLSTM(config)
end

-----------------------------------------------------------------------
---- Init variables and convert model to cuda if flag set.
-----------------------------------------------------------------------
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion())
local clerr = tnt.ClassErrorMeter{topk = {3}}
local timer = tnt.TimeMeter()
local torchTimer = torch.Timer()
local batch = 1
local lastFinishTime = -1
local startTime = -1

if opt.cuda then 
    print("Using CUDA")
    require 'cunn'
    require 'cudnn'
    require 'cutorch'
    cudnn.benchmark = true
    cudnn.fastest = true
    cutorch.setDevice(opt.gpuid)
    cudnn.convert(model, cudnn)

    -- if opt.nGPU > 1 then
    --     print("Using " .. opt.nGPU .. ' GPUs.')
    --     local singlemodel = model:float()
    --     print('copy single model')
    --     collectgarbage()
    --     model = nn.DataParallelTable(1, true, false)
    --         for i = 1, opt.nGPU do
    --             print(i)
    --             cutorch.setDevice(i)
    --             model:add(singlemodel:clone():cuda(), i)
    --             print(i .. ' end')
    --         end
    --     cutorch.setDevice(1)
    -- else 
    --     model:cuda()
    -- end
    model:cuda()
    criterion:cuda()

    engine.hooks.onSample = function(state)
        -- print(state.sample.input.image)
        assert(type(state.sample.input.image) == 'userdata', "incorrect data type: " .. type(state.sample.input.image))
        assert(state.sample.input.image:type() == "torch.DoubleTensor", "image type incorrect, got type: " .. state.sample.input.image:type())
        state.sample.input.image = state.sample.input.image:cuda()
        state.sample.input.text = state.sample.input.text:cuda()
        if state.sample.target then
            state.sample.target = state.sample.target:cuda()
        end
    end

end


-----------------------------------------------------------------------
---- Train with engine
-----------------------------------------------------------------------

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    lastFinishTime = -1
    startTime = torchTimer:time().real
    if state.training then
        mode = 'Train'
        num_iters = math.floor(train_set_size / opt.batchsize)
    else
        mode = 'Val'
        num_iters = math.floor(val_set_size / opt.batchsize)
    end
end

--engine.hooks.onForward = function(state)
--end

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)

    -- train timer and ETA
    local function timePerBatchAndETA()
        if lastFinishTime == -1 then
            lastFinishTime = startTime
        end
        local current = torchTimer:time().real
        local timeElapsed = current - lastFinishTime

        local remainStep = num_iters - batch
        local step = (current - startTime) / batch
        local remainTimeTotal = remainStep * step

        return xlua.formatTime(timeElapsed), xlua.formatTime(remainTimeTotal)
    end

    if opt.verbose == true then
        local timeElapsed, ETA = timePerBatchAndETA()
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f, time: %s, ETA: %s",
            mode, batch, num_iters, meter:value()/opt.batchsize, timeElapsed, ETA))
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end


engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; time: %2.4f",
    mode, meter:value()/opt.batchsize,  timer:value()))
    -- print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    -- mode, meter:value(), clerr:value{k = 1}, timer:value()))
    collectgarbage()
    -- the error on end of the training
     current_loss = meter:value()/opt.batchsize
end

local lr = opt.LR 
local epoch = 1

print("Start Training!")
while epoch <= opt.nEpochs do 
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator('train'),
        optimMethod = optim.adam,
        maxepoch = 1,
        config = {
            learningRate = lr,
            momentum = opt.momentum,
            wd = opt.weightDecay
        }
    }
    cutorch.synchronize()
    trainLoss = meter:value()/opt.batchsize
    engine:test {
        network = model,
        criterion = criterion,
        iterator = getIterator('val')
    }
    cutorch.synchronize()
    valLoss = meter:value()/opt.batchsize

    logger:add{trainLoss, valLoss}

    -- save the model if it is the best so far
    if current_loss <= min_loss then
        print("update model")
        min_loss = current_loss
        cudnn.convert(model, nn)
        torch.save(opt.weightsDir .. opt.model .. '_weights.t7', model:clearState():float())
        cudnn.convert(model, cudnn)
    end

    print('Done with Epoch ' .. tostring(epoch))
    epoch = epoch + 1
end






print("The End!")
