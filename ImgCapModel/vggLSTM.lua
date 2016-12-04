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

    -- load image model after remove the last few layers
    local imageModel = loadcaffe.load(prototxt, binary)

    -- remove the layers after Linear(4096 -> 4096), 43-46
    for i in { 46, 45, 44, 43 } do
        imageModel:remove(i)
    end
    self.imageModel = imageModel

    -- visual rescale layer
    self.visualRescale = nn.Sequential()
            :add(nn.Linear(4096, self.embeddingDim))

    -- language model.
    local LSTMcell = nn.Sequential()
            :add(nn.LSTM(config.embeddingDim, config.embeddingDim, 60))
            :add(nn.Linear(config.embeddingDim, config.num_words))
    self.LSTM = nn.Sequencer(LSTMcell)
end


function vggLSTM:forward(input)
    -- image nets
    self.outputImageModel = self.imageModel:forward(input)
    self.visualFeatureRescaled = self.visualRescale:forward(self.outputImageModel)

    -- vggLSTM -> LSTM -> sequencer -> recursor -> LSTMcell -> nn.LSTM, init h0 with visual feature
    self.LSTM.module.module.modules[1].prevOutput = self.visualFeatureRescaled

    -- LSTM
    self.output = self.LSTM:forward(input)
    return self.output
end

function vggLSTM:backward(input, grad)
    -- backprop the language model
    local gradLSTMInput = self.LSTM:backward(self.visualFeatureRescaled, grad)

    -- backprop the rescale visual feature layer
    local gradVisualFeature = self.visualRescale:backward(self.outputImageModel, gradLSTMInput)
end


