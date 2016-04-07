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
context_string = arg[2]
-- temperature = arg[3]

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

-- sequencer = model:findModules('nn.Sequencer')[2]
-- sequencer:remember()

encoder = model:findModules('nn.Sequential')[3]
-- decoder = model.modules[#model.modules]
-- word_embedder = model:findModules('nn.LookupTable')[1]

-- print(encoder)
-- print(decoder)

-- we'll just use this for the vocab mappings
loader = Loader.create(opt.datasetdir, 'val', 10, 10, 10)

-- inputs = {}
-- for _, word in ipairs(context_string:split(' ')) do
--     table.insert(inputs, loader.word_mappings[word])
-- end


-- input_index = math.random(1, loader.data:size(1))

if context_string then
    input_index = -1
    inputs = {}
    for _, word in ipairs(context_string:split(' ')) do
        table.insert(inputs, loader.word_mappings[word])
    end

    input = torch.Tensor(inputs):reshape(1, #inputs)
else
    input_index = 100

    input, _ = loader:load_batch_table(input_index)
    input = input[1]
    if opt.gpu then
        input = input:cuda()
    end
end

print("Finding the nearest neighbor for:")
for i = 1, input:size(2) do
    io.write(loader.inverse_word_mappings[input[1][i] ], ' ')
end
print('')

encoded_input = encoder:forward(input):clone()

-- [[
nearest_neighbor = nil
nearest_distance = math.huge
for i = 1, loader.data:size(1) do
-- for i = 1, 100 do
    if i ~= input_index then
        -- print("Neighbor: "..i..'/'..loader.data:size(1))
        current_input, _ = loader:load_batch_table(i)
        current_input = current_input[1]
        if opt.gpu then
            current_input = current_input:cuda()
        end

        current_encoding = encoder:forward(current_input)
        -- print('current_encoding: ', current_encoding)
        current_distance = (encoded_input - current_encoding):norm()
        -- print('current_distance: ', current_distance)

        if current_distance < nearest_distance then
            nearest_neighbor = current_input
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
--]]