import pytest
import random, string, functools, json, os, subprocess
import boto3


HOPPER_BENCHMARKING_ACCOUNT = '389009836860'


@pytest.fixture(scope='function')
def function_uid():
    '''
    Returns a unique string specific to a test
    '''
    return ''.join(random.choices(string.ascii_letters + string.digits, k=5))


@pytest.fixture(scope='session')
def session_uid():
    '''
    Returns a unique string specific to a session of tests
    '''
    return ''.join(random.choices(string.ascii_letters + string.digits, k=5))


@pytest.fixture(scope='session')
def credentials(user):
    '''
    Obtains credentials to the testing service
    '''
    sts_client = boto3.client("sts")
    role = sts_client.assume_role(
                    RoleArn=f"arn:aws:iam::{HOPPER_BENCHMARKING_ACCOUNT}:role/StartTestWorkFlow",
                    RoleSessionName=f"{user}-pytorch-benchmarks"
                  )
    return role["Credentials"]


@pytest.fixture(scope='session')
def start_test_exec(credentials):
    '''
    Returns a pre configured call to the testing service
    '''
    sfn_client = boto3.client(
        "stepfunctions",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"]
      )
    return functools.partial(sfn_client.start_execution,
        stateMachineArn=f"arn:aws:states:us-west-2:{HOPPER_BENCHMARKING_ACCOUNT}:stateMachine:EC2_Trainer")


@pytest.fixture(scope='session')
def test_directory():
    '''
    Returns the path to the hopper/test directory
    '''
    hopper_test_dir = os.path.join('hopper','test')
    cwd = os.getcwd()
    parent_dir = None
    if hopper_test_dir in cwd:
        parent_dir = os.path.join(cwd[:cwd.index(hopper_test_dir)], 'hopper', 'test')
        return parent_dir
    else:
        for root, dirs, files in os.walk(cwd):
            if root[len(cwd):].count(os.sep) < 2:
                for d in dirs:
                    if d == 'test' and root.endswith('hopper'):
                        parent_dir = os.path.join(root, 'test')
                        return parent_dir
    assert parent_dir, f"Unable to find directory {hopper_test_dir}. Please try again from the test directory."


@pytest.fixture(scope='session')
def bootstrap(credentials, test_directory, session_uid):
    '''
    Uploads the contents of the hopper/test directory to be bootstrapped into the training container
    by the testing service
    '''
    s3_destination = f"s3://hopper-test-scripts-bootstrapping-{HOPPER_BENCHMARKING_ACCOUNT}-us-west-2/{session_uid}"
    env = os.environ.copy()
    env.update({
                'AWS_ACCESS_KEY_ID':credentials["AccessKeyId"],
                'AWS_SECRET_ACCESS_KEY':credentials["SecretAccessKey"],
                'AWS_SESSION_TOKEN':credentials["SessionToken"],
          })
    bootstrapping = subprocess.run(f"aws s3 cp --recursive --acl bucket-owner-full-control {test_directory} {s3_destination}",
          stderr=subprocess.STDOUT,
          stdout=subprocess.PIPE,
          shell=True,
          env=env,
          )
    assert not bootstrapping.returncode
    return s3_destination


@pytest.fixture(scope='session')
def user():
    '''
    Returns the alias of the user or the AWS service calling the tests
    '''
    arn = boto3.client('sts').get_caller_identity()['Arn']
    user = arn.split('/')[-1].split('-')[0]
    return user


@pytest.fixture(scope='session')
def cwloggroup(user):
    '''
    Default Cloudwatch namespace to upload training logs and metrics
    '''
    return f"Benchmarks/EC2/PyTorch/{user}"


@pytest.fixture(scope='session')
def user_defined_tag():
    return {}


@pytest.fixture(scope='session')
def auto_generated_tags(user):
    '''
    Tagging training resources for cost accountability
    '''
    tags = {}
    tags['User'] = user
    tags['UI'] = "pytest"
    tags['Framework'] = "pytorch"
    tags['Name'] = f"{user}-pytorch-benchmarks"
    return tags


@pytest.fixture(scope='function')
def tags(user_defined_tag, auto_generated_tags):
    '''
    Collates user defined tags and auto generated tags
    '''
    all_tags = auto_generated_tags.copy()
    all_tags.update(user_defined_tag)
    return all_tags


@pytest.fixture(scope='session')
def training_container():
    return None


@pytest.fixture(scope='function')
def _request_template(cwloggroup, tags, training_container, bootstrap):
    '''
    Prefils the request template for the testing service
    '''
    with open("automation/test-template-multiple.json") as f:
        test_config = json.load(f)
    test_config['CloudWatchLogGroupName'] = cwloggroup
    test_config['ResourceConfig']['TagSpecifications'] = [{
                                                            "ResourceType": "instance",
                                                            "Tags": [{'Key':key, 'Value':tags[key]} for key in tags]
                                                        }]
    if training_container:
        test_config['AlgorithmSpecifications']['TrainingImage'] = training_container
    test_config['AlgorithmSpecifications']['TrainingScripts'] = bootstrap
    return test_config


@pytest.fixture(scope='function')
def request_template(_request_template):
    '''
    Returns an unique copy of the testing template
    '''
    return _request_template.copy()


@pytest.fixture(scope='function')
def name_template(session_uid):
    '''
    Returns a unique name for an invocation of the testing service
    '''
    return functools.partial("pt-{session_uid}-{model}-Seq{seq}".format, session_uid=session_uid)



def pytest_addoption(parser):
    parser.addoption('--log', '--cw', '--loggroup', '--cloudwatch', '--cwloggroup', dest='cwloggroup', action="store", help='CloudWatch Log group to upload logs to')
    parser.addoption('--name', '--tag', dest='user_defined_tag', action="store", help='Name to tag benchmarking resources with, for cost accountability')
    parser.addoption('--container', dest='training_containers', action="append", help='List of containers to test')


def pytest_generate_tests(metafunc):
    if "cwloggroup" in metafunc.fixturenames and metafunc.config.getoption("cwloggroup"):
        metafunc.parametrize("cwloggroup", [metafunc.config.getoption("cwloggroup")])
    if "user_defined_tag" in metafunc.fixturenames and metafunc.config.getoption("user_defined_tag"):
        metafunc.parametrize("user_defined_tag", [{'Name':metafunc.config.getoption("user_defined_tag")}])
    if "training_container" in metafunc.fixturenames and metafunc.config.getoption("training_containers"):
        metafunc.parametrize("training_container", metafunc.config.getoption("training_containers"))


