from transformers import BertForMaskedLM
import torch
import torch_xla.core.xla_model as xm
import argparse
import time
import torch_xla
import torch_xla.amp.syncfree as syncfree
import numpy as np
from torch_xla.amp import autocast, GradScaler
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.debug.profiler as xp
import multiprocessing

port_number = 8192

def loop_tb(step, amp, optimizer, scaler, input_ids, attention_mask, labels, pt):
  with xp.StepTrace('train_loop', step_num=step):
    optimizer.zero_grad()
    if amp:
      with xp.Trace('building graph {}'.format(step)):
        with autocast():
          outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
          loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
      with xp.Trace('building graph {}'.format(step)):
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print("forward done {}".format(step))
        loss.backward()
        optimizer.step()

def loop_no_tb(step, amp, optimizer, scaler, input_ids, attention_mask, labels, pt):
  optimizer.zero_grad()
  if amp:
    with autocast():
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
  else:
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
  if not pt:
    xm.mark_step()


def train(device, model, optimizer, batch_size, num_steps, amp=False, pt=False, tb=False):
  input_ids = torch.ones((batch_size, 512)).to(torch.int64).to(device)
  attention_mask = torch.ones((batch_size, 512)).to(torch.int64).to(device)
  labels = torch.ones((batch_size, 512)).to(torch.int64).to(device)
  print("batch_size={}".format(batch_size))
  scaler = GradScaler() if not pt else torch.cuda.amp.GradScaler()
  loop = loop_tb if tb and not pt else loop_no_tb
  for step in range(2):
    loop(step, amp, optimizer, scaler, input_ids, attention_mask, labels, pt)
    print("step={}".format(step))
  torch_xla._XLAC._xla_barrier(torch_xla._XLAC._xla_get_default_device())
  t0 = time.time()
  for step in range(num_steps):
    loop(step, amp, optimizer, scaler, input_ids, attention_mask, labels, pt)
    if step % 10 == 0:
      print("step={}".format(step))
  xm.mark_step()
  t1 = time.time()
  print("speed = {} it/s".format(num_steps / (t1 - t0)))
  print("thrpt = {} samples/s".format(num_steps * batch_size / (t1 - t0)))


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--batch_size", type=int, default=12)
  arg_parser.add_argument("--amp", action='store_true')
  arg_parser.add_argument("--pt", action='store_true')
  arg_parser.add_argument("--tb", action='store_true')
  args = arg_parser.parse_args()
  batch_size = args.batch_size
  amp = args.amp
  pt = args.pt
  tb = args.tb
  device = xm.xla_device() if not pt else torch.device("cuda:0")
  model = BertForMaskedLM.from_pretrained("bert-base-uncased")
  new_config = model.config
  # new_config.num_hidden_layers = 1
  model = BertForMaskedLM(new_config)
  model.to(device)
  model.train()
  # optimizer = AdamW(model.parameters())
  optimizer = syncfree.SGD(model.parameters(), lr=1e-2, momentum=0.9)
  if tb:
    # xp.trace(f"localhost:{port_number}", "tmp/tensorboard")
    server = xp.start_server(port_number)
  train(device, model, optimizer, batch_size, 100, amp, pt, tb)

#   xmp.spawn(train, args=(device, model, optimizer, batch_size, 100, amp), nprocs=1)
