# Hopper

This folder contains files related to builds, testing and CICD.


## Important scripts

 * `hopper/test/scripts/launch_benchmarks.py` - Can be used inside container to trigger all HF benchmarks tests across all combinations of sequence length, batch size, and models.

 * `hopper/test/automation/trigger.py` - Uses Step Function infrastructure to test all benchmarks on EC2 instances.