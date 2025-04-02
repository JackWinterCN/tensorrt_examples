rm -rf build
mkdir build
cd build
cmake ..
cmake --build .
# cp ./main ../
cp ./AveragePluginByAPI ../
cp ./AveragePluginByParser ../