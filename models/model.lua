require 'loadcaffe'
require 'xlua'
require 'optim'
require 'nn'
require 'image'
require 'torch'
require 'rnn'

local prototxt = './VGG_ILSVRC_19_layers_deploy.prototxt'
local binary = './VGG_ILSVRC_19_layers.caffemodel'

local TheModel = {}



function TheModel:__init(inputSize, hiddenSize, batchSize, numIter)
    self.inputSize  = inputSize
    self.hiddenSize = hiddenSize
    self.batchSize = batchSize
    self.numIter = numIter
end



function TheModel:buildModel()
    ------------------------ image module -----------------------
    -- this will load the network and print it's structure
    local image_module = loadcaffe.load(prototxt, binary)

    -- remove the layers after Linear(4096 -> 4096), 43-46
    image_module:remove(46)
    image_module:remove(45)
    image_module:remove(44)
    image_module:remove(43)


    ----------------------- image mapping module -----------------
    local visual_feature = nn.Sequential()
    visual_feature.add(nn.Linear(4096, self.hiddenSize))


    ----------------------- image mapping module -----------------
    local lstm_unit = nn.LSTM(self.hiddenSize, self.hiddenSize) -- use fast or not?

    local rnn = nn.Sequential()
    :add(nn.Linear(self.inputSize, self.hiddenSize))
    :add(lstm_unit)
    :add(nn.NormStabilizer())
    :add(nn.Linear(self.hiddenSize, self.inputSize))
    :add(nn.HardTanh())

    -- let the h0 of the lstm unit share the parameters of the output of the visual feature mapping
    lstm_unit.prevOutput = visual_feature.output

    local lm = nn.Sequencer(rnn)

    ----------------------- add all the three above to the model -----------------
    TheModel.im = image_module
    TheModel.vf = visual_feature
    TheModel.lm = lm
end

function TheModel:forward()

end

function TheModel:backward()

end



return TheModel



