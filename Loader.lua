-- Modified from https://github.com/karpathy/char-rnn

local Loader = {}
Loader.__index = Loader

function Loader.create(data_dir, input_filename, n_context, n_skip, n_predict)
    local self = {}
    setmetatable(self, Loader)

    self.n_context = n_context
    self.n_skip = n_skip
    self.n_predict = n_predict

    seq_length = n_context + n_skip + n_predict

    local input_file = path.join(data_dir, input_filename..'.txt')
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local tensor_file = path.join(data_dir, input_filename..'.t7')


    print('loading data files...')
    local data = torch.load(tensor_file)
    self.vocabulary = torch.load(vocab_file)
    self.word_mappings = self.vocabulary.word_mappings
    self.inverse_word_mappings = self.vocabulary.inverse_word_mappings
    self.vocab_size = #self.inverse_word_mappings

    -- print(self.word_mappings["<PREDICT>"])

    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % seq_length ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data = data:sub(1, seq_length * math.floor(len / seq_length))
    end

    self.data = data:reshape(len / seq_length, seq_length)

    collectgarbage()
    return self
end

function Loader:load_batch(id)
    local batch = self.data[id]
    local x = torch.Tensor(self.n_context + self.n_predict)
    x[{{1, self.n_context}}] = batch[{{1, self.n_context}}]
    x[self.n_context + 1] = self.word_mappings["<PREDICT>"]
    x[{{self.n_context + 2, x:size(1)}}] = batch[{{self.n_context + self.n_skip + 1, batch:size(1) - 1}}]
    local y = batch[{{self.n_context + self.n_skip + 1, batch:size(1)}}]
    -- print(x:size(1))
    x = x:reshape(1, x:size(1))
    -- y = y:reshape(1, y:size(1))
    return x, y
end

function Loader:load_batch_table(id)
    local batch = self.data[id]
    local x = {}

    -- the part that goes to the encoder
    table.insert(x, batch[{{1, self.n_context}}]:reshape(1, self.n_context))

    -- the part that goes to the decoder
    table.insert(x, batch[{{self.n_context + self.n_skip + 1, batch:size(1) - 1}}]:reshape(1, self.n_predict - 1))

    local y = batch[{{self.n_context + self.n_skip + 1, batch:size(1)}}]
    return x, y
end

function Loader:load_random_batch_table()
    return self:load_batch_table(math.random(1, self.data:size(1)))
end

function Loader:load_random_batch()
    return self:load_batch(math.random(1, self.data:size(1)))
end

-- *** STATIC method ***
function Loader.text_to_tensor(in_textfiles, out_vocabfile, out_tensorfiles, min_occurrence)
    min_occurrence = min_occurrence or 5

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')

    local total_words = {}
    local word_counts = {}
    for _, in_textfile in ipairs(in_textfiles) do
        local file_words = 0
        for line in io.lines(in_textfile) do
            for _, word in ipairs(line:split(' ')) do
                file_words = file_words + 1
                if word_counts[word] == nil then
                    word_counts[word] = 1
                else
                    word_counts[word] = word_counts[word] + 1
                end
            end
        end
        table.insert(total_words, file_words)
    end

    local word_mappings = {}
    local inverse_word_mappings = {}
    word_mappings["<PREDICT>"] = 1
    inverse_word_mappings[1] = "<PREDICT>"

    word_mappings["<UNK>"] = 2
    inverse_word_mappings[2] = "<UNK>"

    for word, count in pairs(word_counts) do
        if count >= min_occurrence then
            local next_id = #inverse_word_mappings + 1
            word_mappings[word] = next_id
            inverse_word_mappings[next_id] = word
        end
    end

    print(word_mappings)

    for input_index, in_textfile in ipairs(in_textfiles) do
        -- construct a tensor with all the data
        print('putting data into tensor...')

        -- store it into 1D first, then rearrange
        local data = torch.Tensor(total_words[input_index])

        local current = 1
        for line in io.lines(in_textfile) do
            for _, word in ipairs(line:split(' ')) do
                -- the ID for the word is 1 (for <UNK>) if it's not in the list
                local word_id = word_mappings[word] or word_mappings["<UNK>"]

                data[current] = word_id
                current = current + 1
            end
        end


        -- save output preprocessed files
        print('saving ' .. out_tensorfiles[input_index])
        torch.save(out_tensorfiles[input_index], data)
    end
    local vocabulary = {word_mappings = word_mappings, inverse_word_mappings = inverse_word_mappings}
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocabulary)

end

return Loader
