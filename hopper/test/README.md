# Interface

Testing is handled through PyTest.

You can specify the container(s) to be tested using --container

You can specify the CloudWatch namespace for training logs using --cw

You can tag your resources for cost accountability using --tag

## Defaults

The default container is specified by the test definition. The default cloudwatch namespace is obtained from your credentials. By default all resources are tagged with your credentials as well.

## Bootstrapping

All the contents of hopper/test are bootstrapped from your local working copy into the training container under /opt/ml/code/hopper/test

# Requirements

Tests are executed by a service owned by the Training Compiler team in the hopper-benchmark AWS account. As a customer, you would need to be allow listed by the testing service. Please contact hopper-benchmark@amazon.com to be allowlisted. 

# How do the tests work ?

PyTest uses standard discovery to discover the tests. The benchmarks are executed on EC2 by a service owned by the Training Compiler team. PyTest is only responsible for kicking off the tests. Please check the logs/dashboards in the hopper-benchmark account to verify test results.

# How to add tests ?

To add benchmarks, the training scripts need to be present inside the hopper/test directory. Follow pattern of existing benchmarks in test_performance.py to add the tests for discovery by PyTest.


# How to add tests for Nightly?

We use PyTest marks to group tests. However, marking the tests as Nightly only causes them to be run on a nightly basis. However, the Hopper dashboard is unable to discover these new tests and consequently will not be reported. You also need to make changes to the appropriate Dashboard CDK Stacks in the AWSHopperBenchmarkInfraCDK package for reporting.






