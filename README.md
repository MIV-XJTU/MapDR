# 【CVPR 2025】Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map

**HomePage: [Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map](https://miv-xjtu.github.io/MapDR/)**

## Paper  

ArXiv: [Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map](https://arxiv.org/abs/2410.23780v2)

## Dataset  

Demo dataset : [[ModelScope] MapDR-mini](https://modelscope.cn/datasets/MIV-XJTU/MapDR-mini)

Full datasest : [[ModelScope] MapDR](https://modelscope.cn/datasets/MIV-XJTU/MapDR/)

**Note: All dataset slices contained in ModelScope need to be downloaded in full to synthesize the full dataset.**

## Code  

GitHub: [MapDR](https://github.com/MIV-XJTU/MapDR)

## Unzip the dataset  

```bash
# First concatenate the full dataset
cat mapdr_* > mapdr.tar.gz

# unzip the full dataset
tar xvzf mapdr.tar.gz
```

## Usage of Visualization

First Download MapDR_v1220 or MapDR_mini from Modelscope

```python

# Create a video for target case

# Example : python visualize.py /mapdr_mini/BusLane/0cdea530a3c24022b22a7320ad2e4818 ./visualization

python visualize.py path/to/data path/to/save 

```

## Usage of Evaluation  

TODO

## Citation

TODO
