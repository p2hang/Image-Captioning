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
	self.loadImageModel(config.prototxt, config.binary) -- load the image model

	self.visualFeature = nn.Sequential()
					:add(nn.Linear(4096, self.embeddingDim))

	-- let the LSTM unit share the output of the visual feature after rescale
	local lstmCell = nn.LSTM(config.embeddingDim, config.embeddingDim, 60)
	lstmCell.prevOutput = self.visualFeature.output


	local rnnUnit = nn.Sequential()
					:add(lstmCell)
					:add(nn.Linear(config.embeddingDim, config.num_words))
	self.LSTM = nn.Sequencer(rnnUnit)
end

function vggLSTM:loadImageModel(prototxt, binary)
	local imageModel = loadcaffe.load(prototxt, binary)

	-- remove the layers after Linear(4096 -> 4096), 43-46
	imageModel:remove(46)
	imageModel:remove(45)
	imageModel:remove(44)
	imageModel:remove(43)
	self.imageModel = imageModel
end

function vggLSTM:forward(input)
	-- image nets
	self.image_representation = self.vgg19:forward(input)
	self.
	-- LSTM
	self.output = self.LSTM:forward(self.image_representation)
	return self.output
end

function vggLSTM:backward(input, grad)

end
