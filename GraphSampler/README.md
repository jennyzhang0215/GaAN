# C++ Extensions for Graph Sampler in Python

The sampler of the graph.

The graph is assumed to have this format
- node_types: ...
- end_points: ...
- ind_ptr: ...
- node_ids: ...

The sampled_graph will have this format
- Sample a subgraph from the given graph
- Sample a subgraph from the given graph w.r.t some given nodes

# Install
For windows users:

```bash
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES="Release" ..
```
Open GraphSampler.sln and use VS 2015 to build, then
```bash
cd ..
python install.py
```

For unix users (including macOS and linux):
```bash
mkdir build
cd build
cmake ..
make
cd ..
python install.py
```
