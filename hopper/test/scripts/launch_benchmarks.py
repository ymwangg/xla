import os
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--use-xla", action='store_true')
  parser.add_argument('-m', '--models', nargs='+', default=["bert-base-uncased", "bert-large-uncased", "/opt/ml/code/hopper/test/files/bart-config.json", "roberta-base", "gpt2"])
  parser.add_argument('-s', '--sequence_lengths', nargs='+', type=int, default=[128, 512])
  parser.add_argument('-b', '--batch_sizes', nargs='+', type=int, default=[1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 96, 128])
  parser.add_argument('-d', '--transformers-dir', default="/opt/ml/code/transformers")
  args = parser.parse_args()

  for model in args.models:
    for seq_len in args.sequence_lengths:
      for batch_size in args.batch_sizes:
        print("running {} batch_size={} sequence_length={} with xla={}".format(model, batch_size, seq_len, args.use_xla))
        if args.use_xla:
          os.system("python3 {}/examples/pytorch/benchmarking/run_benchmark.py --models {} --training yes --batch_sizes {} --sequence_lengths {} --inference no --tpu --memory false --fp16".format(
            args.transformers_dir, model, batch_size, seq_len
          ))
        else:
          os.system("python3 {}/examples/pytorch/benchmarking/run_benchmark.py --models {} --training yes --batch_sizes {} --sequence_lengths {} --inference no --memory false --cuda yes --fp16".format(
            args.transformers_dir, model, batch_size, seq_len
          ))
