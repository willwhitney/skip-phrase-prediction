require 'rnn'
require 'Print'


local SkipLSTM = function(vocab_size, hidden_size, n_context, n_predict)
    local model = nn.Sequential()



    model:add(nn.LookupTable(vocab_size, hidden_size))
    model:add(nn.SplitTable(1))

    local lstm = nn.Sequencer(nn.LSTM(hidden_size, hidden_size))
    model:add(lstm)

    model:add(nn.JoinTable(1))
    model:add(nn.Reshape(n_context + n_predict, hidden_size))
    -- model:add(nn.Print('after LSTM'))
    -- encoder:add(nn.SelectTable(-1))


    -- local decoder = nn.Sequential()

    -- local decoder_lstm = nn.Sequencer(nn.LSTM(hidden_size, hidden_size))
    -- decoder:add(decoder_lstm)
    model:add(nn.Narrow(1, n_context + 1, n_predict))
    model:add(nn.Linear(hidden_size, vocab_size))
    model:add(nn.LogSoftMax())

    return model
end

return SkipLSTM
