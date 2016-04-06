require 'rnn'
require 'Print'


local SkipLSTM = function(vocab_size, hidden_size, n_context, n_predict)
    local model = nn.Sequential()
    local block1 = nn.Sequential()

    block1:add(nn.LookupTable(vocab_size, hidden_size))
    block1:add(nn.SplitTable(2))

    local lstm = nn.Sequencer(nn.LSTM(hidden_size, hidden_size))
    block1:add(lstm)

    -- block1:add(nn.Print('after LSTM'))
    -- block1:add(nn.Reshape(1, hidden_size))
    block1:add(nn.JoinTable(1))
    -- block1:add(nn.Print('after join'))

    model:add(block1)
    -- model:add(nn.Reshape(n_context + n_predict, hidden_size))
    -- encoder:add(nn.SelectTable(-1))


    -- local decoder = nn.Sequential()

    -- local decoder_lstm = nn.Sequencer(nn.LSTM(hidden_size, hidden_size))
    -- decoder:add(decoder_lstm)
    local block2 = nn.Sequential()
    block2:add(nn.Narrow(1, n_context + 1, n_predict))
    block2:add(nn.Linear(hidden_size, vocab_size))
    block2:add(nn.LogSoftMax())
    -- block2:add(nn.Print('after softmax'))
    model:add(block2)
    return model
end

return SkipLSTM
