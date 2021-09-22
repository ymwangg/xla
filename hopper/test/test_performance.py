import json
import pytest



@pytest.mark.nightly
@pytest.mark.huggingface
class TestPyTorchHuggingFacePerformance:
    @pytest.mark.huggingface_top5
    @pytest.mark.bert
    def test_BertBaseUncased_performance_for_Seq512(self, start_test_exec, request_template, name_template):
        model, seq, batch = "bert-base-uncased", str(512), [16, 24]
        request_template["HyperParameters"]["models"] = model
        request_template["HyperParameters"]["sequence_lengths"] = seq
        request_template["HyperParameters"]["batch_sizes"] = " ".join(map(str, batch))
        name = name_template(model=model, seq=seq)
        response = start_test_exec( input=json.dumps(request_template),
                                    name=name,
                                )


    @pytest.mark.huggingface_top5
    @pytest.mark.bert
    def test_BertLargeUncased_performance_for_Seq512(self, start_test_exec, request_template, name_template):
        model, seq, batch = "bert-large-uncased", str(512), [4, 8]
        request_template["HyperParameters"]["models"] = model
        request_template["HyperParameters"]["sequence_lengths"] = seq
        request_template["HyperParameters"]["batch_sizes"] = " ".join(map(str, batch))
        name = name_template(model=model, seq=seq)
        response = start_test_exec( input=json.dumps(request_template),
                                    name=name,
                                )


    @pytest.mark.huggingface_top5
    @pytest.mark.roberta
    def test_RobertaBase_performance_for_Seq512(self, start_test_exec, request_template, name_template):
        model, seq, batch = "roberta-base", str(512), [12, 20]
        request_template["HyperParameters"]["models"] = model
        request_template["HyperParameters"]["sequence_lengths"] = seq
        request_template["HyperParameters"]["batch_sizes"] = " ".join(map(str, batch))
        name = name_template(model=model, seq=seq)
        response = start_test_exec( input=json.dumps(request_template),
                                    name=name,
                                )


    @pytest.mark.huggingface_top5
    @pytest.mark.gpt2
    def test_Gpt2_performance_for_Seq512(self, start_test_exec, request_template, name_template):
        model, seq, batch = "gpt2", str(512), [8, 24]
        request_template["HyperParameters"]["models"] = model
        request_template["HyperParameters"]["sequence_lengths"] = seq
        request_template["HyperParameters"]["batch_sizes"] = " ".join(map(str, batch))
        name = name_template(model=model, seq=seq)
        response = start_test_exec( input=json.dumps(request_template),
                                    name=name,
                                )


    @pytest.mark.huggingface_top5
    @pytest.mark.bart
    def test_Bart_performance_for_Seq512(self, start_test_exec, request_template, name_template):
        model, seq, batch = "/opt/ml/code/hopper/test/files/bart-base.json", str(512), [20, 32]
        request_template["HyperParameters"]["models"] = model
        request_template["HyperParameters"]["sequence_lengths"] = seq
        request_template["HyperParameters"]["batch_sizes"] = " ".join(map(str, batch))
        name = name_template(model='bart', seq=seq)
        response = start_test_exec( input=json.dumps(request_template),
                                    name=name,
                                )

