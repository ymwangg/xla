import boto3
import json
import random
import string

def get_random_string(length=6):
  return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

if __name__ == "__main__":
  client = boto3.client('stepfunctions')
  # Load template
  with open("test-template-multiple.json") as f:
    test_config = json.load(f)
  # Launch new execution for each model, seq_len
  # Each execution will do all batch sizes for that model, seq_len
  models = ["bert-base-uncased", "bert-large-uncased", "/opt/ml/code/hopper/test/files/bart-config.json", "roberta-base", "gpt2"]
  for model in models:
    for seq_len in [128, 512]:
      test_config["HyperParameters"]["models"] = model
      test_config["HyperParameters"]["sequence_lengths"] = str(seq_len)
      response = client.start_execution(
        stateMachineArn='arn:aws:states:us-west-2:886656810413:stateMachine:Benchmarking_EC2_Training',
        name='hopperbench-{}-{}-{}'.format(model.split("/")[-1], seq_len, get_random_string()),
        input=json.dumps(test_config)
      )
      print(response)
      # TODO: Collect CommandId from jobs and retrieve cloudwatch logs
