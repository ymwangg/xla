import pytest
import random, string, functools, json
import boto3


HOPPER_BENCHMARKING_ACCOUNT = '389009836860'


@pytest.fixture(scope='function')
def function_uid():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=5))


@pytest.fixture(scope='session')
def session_uid():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=5))


@pytest.fixture(scope='session')
def start_test_exec():
    sts_client = boto3.client("sts")
    role = sts_client.assume_role(
                    RoleArn=f"arn:aws:iam::{HOPPER_BENCHMARKING_ACCOUNT}:role/StartTestWorkFlow",
                    RoleSessionName="AssumeRoleSession"
                  )
    credentials = role["Credentials"]
    sfn_client = boto3.client(
        "stepfunctions",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"]
      )
    return functools.partial(sfn_client.start_execution,
        stateMachineArn=f"arn:aws:states:us-west-2:{HOPPER_BENCHMARKING_ACCOUNT}:stateMachine:EC2_Trainer")


@pytest.fixture(scope='session')
def user():
    arn = boto3.client('sts').get_caller_identity()['Arn']
    user = arn.split('/')[-1].replace('-Isengard','')
    return user


@pytest.fixture(scope='session')
def cwloggroup(user):
    return f"Benchmarks/EC2/PyTorch/{user}"


@pytest.fixture(scope='session')
def user_defined_tag():
    return {}


@pytest.fixture(scope='session')
def auto_generated_tags(user):
    tags = {}
    tags['User'] = user
    tags['UI'] = "pytest"
    tags['Framework'] = "pytorch"
    tags['Name'] = f"{user}-pytorch-benchmarks"
    return tags


@pytest.fixture(scope='function')
def tags(user_defined_tag, auto_generated_tags):
    all_tags = auto_generated_tags.copy()
    all_tags.update(user_defined_tag)
    return all_tags


@pytest.fixture(scope='session')
def training_container():
    return None


@pytest.fixture(scope='function')
def _request_template(cwloggroup, tags, training_container):
    with open("automation/test-template-multiple.json") as f:
        test_config = json.load(f)
    test_config['CloudWatchLogGroupName'] = cwloggroup
    test_config['ResourceConfig']['TagSpecifications'] = [{
                                                            "ResourceType": "instance",
                                                            "Tags": [{'Key':key, 'Value':tags[key]} for key in tags]
                                                        }]
    if training_container:
        test_config['AlgorithmSpecifications']['TrainingImage'] = training_container
    return test_config


@pytest.fixture(scope='function')
def request_template(_request_template):
    return _request_template.copy()


@pytest.fixture(scope='function')
def name_template(session_uid):
    return functools.partial("hopper-{session_uid}-{model}-Seq{seq}".format, session_uid=session_uid)



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


