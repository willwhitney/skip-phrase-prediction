require 'nn'
require 'optim'

require 'cutorch'
require 'cunn'

SkipEncoderDecoder = require 'SkipEncoderDecoder'
Loader = require 'Loader'

-- local seed = torch.random()
-- print("Random seed: ", seed)
-- math.randomseed(seed)

network_name = arg[1]

base_directory = 'networks'

function getLastSnapshot(name)
    res_file = io.popen("ls -t "..paths.concat(base_directory, name).." | grep -i epoch | head -n 1")
    result = res_file:read():match( "^%s*(.-)%s*$" )
    res_file:close()
    return result
end

last_snapshot = getLastSnapshot(network_name)
checkpoint = torch.load(path.join(base_directory, network_name, last_snapshot))
opt = checkpoint.opt

model = checkpoint.model


encoder = model:findModules('nn.Sequential')[3]
loader = Loader.create(opt.datasetdir, 'train', 10, 10, 10)


function get_input(idx)
    local phrase, _ = loader:load_batch_table(idx)
    phrase = phrase[1]
    if opt.gpu then
        phrase = phrase:cuda()
    end
    return phrase
end

function build_embedding_list()
    local embeddings = {}
    for i = 1, math.min(100000, loader.data:size(1)) do
    -- for i = 1, 100 do
        current_input = get_input(i)
        table.insert(embeddings, encoder:forward(current_input):clone())
    end
    return embeddings
end

local embeddings = build_embedding_list()

-- if context_string then
--     input_index = -1
--     inputs = {}
--     for _, word in ipairs(context_string:split(' ')) do
--         table.insert(inputs, loader.word_mappings[word])
--     end

--     input = torch.Tensor(inputs):reshape(1, #inputs)
-- else
--     input_index = 100

--     input, _ = loader:load_batch_table(input_index)
--     input = input[1]
--     if opt.gpu then
--         input = input:cuda()
--     end
-- end

function embed_string(s)
    local inputs = {}
    for _, word in ipairs(s:split(' ')) do
        table.insert(inputs, loader.word_mappings[word])
    end

    local string_tensor = torch.Tensor(inputs):reshape(1, #inputs)

    return string_tensor, encoder:forward(string_tensor):clone()
end

function distance(i1, i2)
    if type(i1) == 'number' then
        encoded_i1 = embeddings[i1]
        i1 = get_input(i1)
    else
        i1, encoded_i1 = embed_string(i1)
    end
    if type(i2) == 'number' then
        encoded_i2 = embeddings[i2]
        i2 = get_input(i2)
    else
        i2, encoded_i2 = embed_string(i2)
    end

    print("Finding the distance between:")
    for i = 1, i1:size(2) do
        io.write(loader.inverse_word_mappings[i1[1][i] ], ' ')
    end
    print('')

    print("and:")
    for i = 1, i2:size(2) do
        io.write(loader.inverse_word_mappings[i2[1][i] ], ' ')
    end
    print('')

    return (encoded_i1 - encoded_i2):norm()
end


-- encoded_input = encoder:forward(input):clone()

function find_nearest_neighbor(input)
    if type(input) == 'number' then
        input_index = input
        encoded_input = embeddings[input]
        input = get_input(input)
    else
        input, encoded_input = embed_string(input)
        input_index = -1
    end

    print("Finding the nearest neighbor for:")
    for i = 1, input:size(2) do
        io.write(loader.inverse_word_mappings[input[1][i] ], ' ')
    end
    print('')

    nearest_neighbor = nil
    nearest_distance = math.huge
    for i = 1, #embeddings do
        if i ~= input_index then
            current_encoding = embeddings[i]
            current_distance = (encoded_input - current_encoding):norm()

            if current_distance < nearest_distance then
                nearest_neighbor = get_input(i)
                nearest_distance = current_distance
            end
        end
    end

    print("Input phrase:")
    for i = 1, input:size(2) do
        io.write(loader.inverse_word_mappings[input[1][i] ], ' ')
    end
    print('')

    print("Nearest neighbor:")
    for i = 1, nearest_neighbor:size(2) do
        io.write(loader.inverse_word_mappings[nearest_neighbor[1][i] ], ' ')
    end
    print('')
end


--]]
