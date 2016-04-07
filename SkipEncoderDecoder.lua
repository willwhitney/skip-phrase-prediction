require 'rnn'
require 'Print'


local SkipEncoderDecoder = function(vocab_size, hidden_size, layers)
    local model = nn.Sequential()

    local word_embedder = nn.LookupTable(vocab_size, hidden_size)

    local encoder = nn.Sequential()
    encoder:add(word_embedder)
    encoder:add(nn.SplitTable(2))

    for _ = 1, layers do
        local encoder_lstm = nn.Sequencer(nn.LSTM(hidden_size, hidden_size))
        encoder:add(encoder_lstm)
    end
    encoder:add(nn.SelectTable(-1))

    local input_block = nn.Sequential()
    local input_parallel = nn.ParallelTable()
        input_parallel:add(encoder)

        local predicted_embedder = nn.Sequential()
            predicted_embedder:add(word_embedder:clone('weight', 'gradWeight'))
            predicted_embedder:add(nn.SplitTable(2))
        input_parallel:add(predicted_embedder)
    input_block:add(input_parallel)
    input_block:add(nn.FlattenTable())

    local decoder = nn.Sequential()

    for _ = 1, layers do
        local decoder_lstm = nn.Sequencer(nn.LSTM(hidden_size, hidden_size))
        decoder:add(decoder_lstm)
    end
    decoder:add(nn.JoinTable(1))
    -- decoder:add(nn.Print("Before linear"))
    decoder:add(nn.Linear(hidden_size, vocab_size))
    decoder:add(nn.LogSoftMax())

    model:add(input_block)
    model:add(decoder)

    return model
end

return SkipEncoderDecoder
