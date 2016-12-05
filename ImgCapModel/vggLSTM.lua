require('nn')
require('nngraph')
require('rnn')
require 'loadcaffe'
-- require "util/load_data"
local vggLSTM, parent = torch.class('ImgCap.vggLSTM', 'nn.Module')

function vggLSTM:__init(config)

    self.vocab = ld:loadVocab()
    self.ivocab = ld:loadiVocab()
    self.num_words = #self.ivocab
    self.embedding_vec = nn.LookupTable(self.num_words, config.embeddingDim)
    self.embedding_vec.weight:copy(ld:loadVocab2Emb())

    -- Load image model after remove the last few layers.
    -- loadcaffe [https://github.com/szagoruyko/loadcaffe]
    -- And remove the layers after last linear layer. For vgg last is 42, (4096 -> 4096)
    local imageModel = loadcaffe.load(config.imageModelPrototxt, config.imageModelBinary)
    for i = #imageModel.modules, config.imageOutputLayer + 1, -1 do
        imageModel:remove(i)
    end

    self.imageModel = imageModel


    -- visual rescale layer
    local imageOutputSize = (#self.imageModel.modules[config.imageOutputLayer].weight)[1]
    -- print(imageOutputSize)
    self.visualRescale = nn.Linear(imageOutputSize, config.embeddingDim)


    -- language model.
    LSTMcell = nn.Sequential()
            :add(nn.LSTM(config.embeddingDim, config.embeddingDim, 60))
            :add(nn.Linear(config.embeddingDim, self.num_words))
    print(torch.isTypeOf(LSTMcell, 'nn.Module'))
    self.LSTM = nn.Sequencer(LSTMcell)
end


function vggLSTM:forward(input)
    local inputImage = input.image 
    local inputText = input.caption
    -- image nets
    self.outputImageModel = self.imageModel:forward(inputImage)
    self.visualFeatureRescaled = self.visualRescale:forward(self.outputImageModel)

    -- vggLSTM -> LSTM -> sequencer -> recursor -> LSTMcell -> nn.LSTM, init h0 with visual feature
    self.LSTM.module.module.modules[1].userPrevOutput = self.visualFeatureRescaled

    -- LSTM
    self.output = self.LSTM:forward(inputText)
    return self.output
end

function vggLSTM:backward(input, grad)
    local inputImage = input.image 
    local inputText = input.caption
    -- backprop the language model
    local gradLSTMInput = self.LSTM:backward(inputText, grad)

    -- backprop the rescale visual feature layer
    local gradVisualFeature = self.visualRescale:backward(self.outputImageModel, gradLSTMInput)
end




function vggLSTM:parameters()
    local modules = nn.Parallel()
    modules:add(self.embedding_vec)
           :add(self.imageModel)
           :add(self.LSTM)
    return modules:parameters()
end

return vggLSTM