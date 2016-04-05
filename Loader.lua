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
    x[self.n_context + 1] = 0
    x[{{self.n_context + 2, x:size(1)}}] = batch[{{self.n_context + self.n_skip + 1, batch:size(1) - 1}}]
    local y = batch[{{self.n_context + self.n_skip + 1, batch:size(1)}}]
    return x, y
end

function Loader:load_random_batch()
    return self:load_batch(math.random(1, self.data:size(1)))
end

-- *** STATIC method ***
function Loader.text_to_tensor(in_textfiles, out_vocabfile, out_tensorfiles, min_occurrence)
    min_occurrence = min_occurrence or 5

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')

    local total_words = 0
    local word_counts = {}
    for _, in_textfile in ipairs(in_textfiles) do
        for line in io.lines(in_textfile) do
            for _, word in ipairs(line:split(' ')) do
                total_words = total_words + 1
                if word_counts[word] == nil then
                    word_counts[word] = 1
                else
                    word_counts[word] = word_counts[word] + 1
                end
            end
        end
    end

    local word_mappings = {}
    local inverse_word_mappings = {}
    word_mappings["<UNK>"] = 1
    inverse_word_mappings[1] = "<UNK>"

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
        local data = torch.Tensor(total_words) -- store it into 1D first, then rearrange

        local current = 1
        for line in io.lines(in_textfile) do
            for _, word in ipairs(line:split(' ')) do
                -- the ID for the word is 1 (for <UNK>) if it's not in the list
                local word_id = word_mappings[word] or 1

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
