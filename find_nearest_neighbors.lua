require 'nn'
require 'optim'

require 'cutorch'
require 'cunn'

local SkipEncoderDecoder = require 'SkipEncoderDecoder'
local Loader = require 'Loader'

local network_name = arg[1]
-- local context_string = arg[2]
-- local temperature = arg[3]

local base_directory = 'networks'

function getLastSnapshot(name)
    local res_file = io.popen("ls -t "..paths.concat(base_directory, name).." | grep -i epoch | head -n 1")
    local result = res_file:read():match( "^%s*(.-)%s*$" )
    res_file:close()
    return result
end

local last_snapshot = getLastSnapshot(network_name)
local checkpoint = torch.load(path.join(base_directory, network_name, last_snapshot))
opt = checkpoint.opt

local model = checkpoint.model

-- local sequencer = model:findModules('nn.Sequencer')[2]
-- sequencer:remember()

local encoder = model:findModules('nn.Sequential')[3]
-- local decoder = model.modules[#model.modules]
-- local word_embedder = model:findModules('nn.LookupTable')[1]

-- print(encoder)
-- print(decoder)

-- we'll just use this for the vocab mappings
local loader = Loader.create(opt.datasetdir, 'val', opt.n_context, opt.n_skip, opt.n_predict)

-- local inputs = {}
-- for _, word in ipairs(context_string:split(' ')) do
--     table.insert(inputs, loader.word_mappings[word])
-- end


local input_index = math.random(1, loader.data:size(1))
local input, _ = loader:load_batch_table(input_index)
if opt.gpu then
    input = {
            input[1]:cuda(),
            input[2]:cuda(),
        }
end

print(input)

local encoded_input = encoder:forward(input)

local nearest_neighbor = nil
local nearest_distance = math.huge
for i = 1, loader.data:size(1) do
    if not i == input_index then
        local current_input, _ = loader:load_batch_table(i)
        if opt.gpu then
            current_input = {
                    current_input[1]:cuda(),
                    current_input[2]:cuda(),
                }
        end

        current_encoding = encoder:forward(input)
        current_distance = (encoded_input - current_encoding):norm()

        if current_distance < nearest_distance then
            nearest_neighbor = current_input
            nearest_distance = current_distance
        end
    end
end

print("Input phrase:")
for i = 1, input:size(1) do
    print(loader.inverse_word_mappings[input[i]])
end

print("Nearest neighbor:")
for i = 1, nearest_neighbor:size(1) do
    print(loader.inverse_word_mappings[nearest_neighbor[i]])
end
