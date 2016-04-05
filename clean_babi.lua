-- print(arg)

local input_name = arg[1]
local output_name = arg[2]

local output_file = io.open(output_name, 'w')

for line in io.lines(input_name) do
    local stripped = line:match( "^%s*(.-)%s*$" )
    if stripped:len() > 0 and stripped:sub(1, 2) ~= '21' then
        output_file:write(stripped .. '\n')
    end
end
