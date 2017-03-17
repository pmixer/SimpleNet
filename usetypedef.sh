# replace struct xxx with xxx using typedef, more concise
# Interesting, run this shell and itself would be modified lol
sed -i "s/struct Vector/Vector/g" `grep -rl 'struct Vector'`
sed -i "s/struct Mat/Mat/g" `grep -rl 'struct Mat'`
sed -i "s/struct SimpleNet/SimpleNet/g" `grep -rl 'struct SimpleNet'`
sed -i "s/struct InputLayer/InputLayer/g" `grep -rl 'struct InputLayer'`
sed -i "s/struct ConnectionLayer/ConnectionLayer/g" `grep -rl 'struct ConnectionLayer'`
sed -i "s/struct TransformLayer/TransformLayer/g" `grep -rl 'struct TransformLayer'`
