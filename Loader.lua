
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

local Loader = {}
Loader.__index = Loader

function Loader.create(data_dir, n_context, n_skip, n_predict)
    local self = {}
    setmetatable(self, Loader)

    self.n_context = n_context
    self.n_skip = n_skip
    self.n_predict = n_predict

    seq_length = n_context + n_skip + n_predict

    local input_file = path.join(data_dir, 'input.txt')
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local tensor_file = path.join(data_dir, 'data.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_file) or path.exists(tensor_file)) then
        -- prepro files do not exist, generate them
        print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        local vocab_attr = lfs.attributes(vocab_file)
        local tensor_attr = lfs.attributes(tensor_file)
        if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
            print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        Loader.text_to_tensor(input_file, vocab_file, tensor_file)
    end

    print('loading data files...')
    local data = torch.load(tensor_file)
    self.vocabulary = torch.load(vocab_file)
    self.word_mappings = self.vocabulary.word_mappings
    self.inverse_word_mappings = self.vocabulary.inverse_word_mappings

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

function Loader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
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

-- *** STATIC method ***
function Loader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile, min_occurrence)
    min_occurrence = min_occurrence or 5

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')

    local total_words = 0
    local word_counts = {}
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

    local word_mappings = {}
    local inverse_word_mappings = {}
    word_mappings["<UNK>"] = 1
    inverse_word_mappings[1] = "<UNK>"

    for word, count in pairs(word_counts) do
        if count >= min_occurrence then
            local next_id = #word_mappings + 1
            word_mappings[word] = next_id
            inverse_word_mappings[next_id] = word
        end
    end

    -- construct a tensor with all the data
    print('putting data into tensor...')
    local data = torch.Tensor(total_words) -- store it into 1D first, then rearrange

    local current = 1
    for line in io.lines(in_textfile) do
        for _, word in ipairs(line:split(' ')) do
            data[current] = word_mappings[word]
        end
    end

    local vocabulary = {word_mappings = word_mappings, inverse_word_mappings = inverse_word_mappings}
    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocabulary)
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data)
end

return Loader
