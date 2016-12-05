require('nn')
require('nngraph')
require('rnn')
require 'loadcaffe'
local vggLSTM, parent = torch.class('ImgCap.vggLSTM', 'nn.Module')

function vggLSTM:__init(config)

    self.vocab = ld:loadVocab()
    self.ivocab = ld:loadiVocab()
    self.num_words = #ivocab
    self.embedding_vec = nn.LookupTable(config.num_words, config.embeddingDim)
    self.embedding_vec.weight:copy(ld:loadVocab2Emb())

    -- Load image model after remove the last few layers.
    -- loadcaffe [https://github.com/szagoruyko/loadcaffe]
    -- And remove the layers after last linear layer. For vgg last is 42, (4096 -> 4096)
    local imageModel = loadcaffe.load(config.prototxt, config.binary)
    for i = #imageModel.modules, config.imageOutputLayer + 1, -1 do
        imageModel:remove(i)
    end
    self.imageModel = imageModel


    -- visual rescale layer
    local imageOutputSize = (#imageModel.modules[config.imageOutputLayer].weight)[1]
    self.visualRescale = nn.Sequential()
            :add(nn.Linear(imageOutputSize, self.embeddingDim))


    -- language model.
    local LSTMcell = nn.Sequential()
            :add(nn.LSTM(config.embeddingDim, config.embeddingDim, 60))
            :add(nn.Linear(config.embeddingDim, config.num_words))

    self.LSTM = nn.Sequencer(LSTMcell)
end


function vggLSTM:forward(inputImage, inputText)
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
    -- backprop the language model
    local gradLSTMInput = self.LSTM:backward(self.visualFeatureRescaled, grad)

    -- backprop the rescale visual feature layer
    local gradVisualFeature = self.visualRescale:backward(self.outputImageModel, gradLSTMInput)
end

return vggLSTM


