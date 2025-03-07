import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = 'data'
TOKENIZER_MODEL = "./data/tok4096.model"


def process_shard(args, vocab_size, tokenizer_model_path):
    """
    处理数据分片，将其中的文本进行分词并保存为二进制文件。

    参数:
    args: tuple, 包含分片ID和分片文件名
    vocab_size: int, 词汇表大小，用于决定输出文件存储路径
    tokenizer_model_path: str, 分词器模型路径
    """
    shard_id, shard = args

    # 根据词汇表大小确定输出文件名
    if vocab_size == 0:
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)  # 确保目录存在
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)

    # 检查文件是否已存在，如果存在则直接返回
    if os.path.exists(tokenized_filename):
        print(f"{tokenized_filename} already exists, skipping.")
        return

    # 初始化分词器
    enc = Tokenizer(tokenizer_model_path)

    # 打开并读取当前分片的JSON文件
    with open(shard, "r") as f:
        data = json.load(f)

    all_tokens = []

    for example in tqdm(data, position=shard_id, desc=f"Shard {shard_id}"):  # Added desc
        text = example["story"]
        text = text.strip()

        tokens = enc.encode(text, bos=True, eos=False)
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.uint16)

    # 将token以二进制形式保存
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


# 定义预处理函数，用于对多个数据分片进行批量处理
def pretokenize(vocab_size):
    """
    预处理所有的数据分片，并将分词后的数据保存为二进制文件。
    
    参数:
    vocab_size: int, 词汇表大小，用于决定输出文件存储路径
    """
    # 数据所在目录
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    
    # 获取所有JSON文件的文件名列表，并按字典序排序
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    
    # 如果词汇表大小大于0，则创建对应的保存目录
    if vocab_size > 0:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # 使用partial函数将vocab_size绑定到process_shard函数
    fun = partial(process_shard, vocab_size=vocab_size, tokenizer_model_path=TOKENIZER_MODEL)
    
    # 使用进程池并行处理每个分片
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """从磁盘加载已预处理的分词数据，并将其以 PyTorch 张量的形式返回。"""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        """
        初始化数据集。

        参数:
        split: str, 数据集的分割方式（'train' 或 'test'）。
        max_seq_len: int, 最大序列长度，用于生成输入输出序列。
        vocab_size: int, 词汇表的大小。
        vocab_source: str, 词汇表的来源（'llama2' 或 'custom'）。
        """
        super().__init__()
        self.split = split  # 数据集划分（训练集或测试集）
        self.max_seq_len = max_seq_len  # 最大序列长度
        self.vocab_size = vocab_size  # 词汇表大小
        self.vocab_source = vocab_source  # 词汇表来源

    def __iter__(self):
        """
        返回迭代器，按批次加载数据并生成模型输入/输出。
        """
        # 获取DataLoader的worker信息（用于并行数据加载）
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0  # worker ID
        # 获取分布式训练的rank信息（用于多GPU训练）
        rank = dist.get_rank() if dist.is_initialized() else 0
        # 基于worker_id和rank生成唯一的随机数种子，确保数据在每个worker和rank之间是唯一的
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")

        # 根据词汇表来源决定数据路径
        if self.vocab_source == "llama2":
            # 如果使用 Llama 2 词汇表，.bin 文件和 .json 文件在同一目录下
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            # 如果使用自定义词汇表，.bin 文件在 tok{N} 目录下
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # 根据数据集划分使用不同的分片文件
        # 训练集使用所有分片文件，测试集只使用第一个分片
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames) > 0, f"在 {bin_dir} 中未找到任何 .bin 文件"

        while True:
            # 随机打乱分片文件
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # 使用 memmap 读取文件，使得数据留在磁盘上，减少内存占用
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                # 计算该分片中的批次数量
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # 去掉最后一个不完整的批次
                assert num_batches > 0, "这个分片文件太小了？请检查。"
                # 随机打乱批次索引
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                # 对每个批次生成输入 x 和目标输出 y
                for ix in ixs:
                    start = ix * self.max_seq_len  # 批次起始索引
                    end = start + self.max_seq_len + 1  # 批次结束索引
                    # 将数据转换为 NumPy 数组并拷贝到 RAM 中
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    # 模型输入 x 是当前批次的前 max_seq_len 个词元
                    x = chunk[:-1]
                    # 模型输出 y 是下一个词元
                    y = chunk[1:]
                    # 生成 x, y 对
                    yield x, y


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    pretokenize(vocab_size=4096)