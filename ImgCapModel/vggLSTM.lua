require('nn')
require('nngraph')
require('rnn')
local vggLSTM, parent = torch.class('ImgCap.vggLSTM', 'nn.Module')

function vggLSTM:__init(config)

	self.vocab = ld:loadVocab()
	self.ivocab = ld:loadiVocab()
	self.num_words = #ivocab
	self.embedding_vec = nn.LookupTable(config.num_words, config.embeddingDim)
	self.embedding_vec.weight:copy(ld:loadVocab2Emb())
	self.vgg19 = -- load vgg
	local LSTMcell = nn.Sequential()
					:add(nn.LSTM(config.embeddingDim, config.embeddingDim, 60))
					:add(nn.Linear(config.embeddingDim, config.num_words))
	self.LSTM = nn.Sequencer(LSTMcell)
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