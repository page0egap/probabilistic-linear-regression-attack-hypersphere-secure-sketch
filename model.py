import torch
import numpy as np

class BackBone(torch.nn.Module):
    def __init__(self, platform, backbone_path, **kwargs):
        super().__init__()
        if platform == "pytorch":
            from insightface.recognition.arcface_torch.backbones import get_model
            model = get_model(**kwargs)
            model.load_state_dict(torch.load(backbone_path))
            model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            model.eval()
            self.model = PytorchModel(model)
        elif platform == "onnx":
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            # set session options do not show warning message
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.log_severity_level = 3
            session = ort.InferenceSession(backbone_path, providers=providers, sess_options=options)
            self.model = ORTModel(session)
        else:
            raise ValueError("platform should be pytorch or onnx")
        
    def forward(self, x:torch.Tensor):
        return self.model(x)
    

class PytorchModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        self.device = next(self.backbone.parameters()).device

    def forward(self, x:torch.Tensor):
        if x.device != self.device:
            x = x.to(self.device)
        result: torch.Tensor = self.backbone(x)
        return result.cpu().numpy()
    

class ORTModel(torch.nn.Module):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.__config()
        self.DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
        self.DEVICE_INDEX = 0
        
    def __config(self):
        input_config = self.session.get_inputs()[0]
        self.input_name = input_config.name
        outputs = self.session.get_outputs()
        output_names = [output.name for output in outputs]
        assert len(output_names) == 1, "Only support single output model"
        self.output_name = output_names[0]

    def forward(self, x:torch.Tensor):
        # x as torch tensor input to session, output as numpy array
        # use onnx runtime binding to avoid copy data between cpu and gpu
        # x as input should be on gpu, the output will be on cpu  
        binding = self.session.io_binding()
        binding.bind_input(
            name=self.input_name,
            device_type=self.DEVICE_NAME,
            device_id=self.DEVICE_INDEX,
            element_type=np.float32,
            shape=x.shape,
            buffer_ptr=x.data_ptr()
        )
        binding.bind_output(name=self.output_name, device_type="cpu")
        self.session.run_with_iobinding(binding)
        result = binding.get_outputs()[0].numpy()
        return result