import torch
import tqdm

from legodnn.common.utils import get_model_flops_and_params, get_model_latency, get_model_size
from ..model_manager.abstract_model_manager import AbstractModelManager


class CommonModelManager(AbstractModelManager):
    def forward_to_gen_mid_data(self, model, batch_data, device):                           #模型基于给定数据集进行一次推理
        model = model.to(device)
        data = batch_data[0].to(device)
        model.eval()
        with torch.no_grad():                                                               #？ 不记录梯度地进行运算  model(data)用法？存疑
            model(data)
            
    def dummy_forward_to_gen_mid_data(self, model, model_input_size, device):
        batch_data = (torch.rand(model_input_size).to(device), None)
        self.forward_to_gen_mid_data(model, batch_data, device)
    
    def get_model_acc(self, model, test_loader, device):
        class AverageMeter:
            def __init__(self):
                self.reset()

            def reset(self):
                self.val = 0
                self.avg = 0
                self.sum = 0
                self.count = 0

            def update(self, val, n=1):
                self.val = val
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count
                
        def _accuracy(output, target, topk=(1,)):
            with torch.no_grad():
                maxk = max(topk)
                batch_size = target.size(0)

                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))

                res = []
                for k in topk:
                    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                    res.append(correct_k.mul_(1. / batch_size))
                return res

        model.eval()
        
        avg_top1_acc_meter = AverageMeter()
        
        with torch.no_grad():
            for i, (data, target) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False):
                data, target = data.to(device, dtype=data.dtype, non_blocking=False, copy=False), \
                    target.to(device, dtype=target.dtype, non_blocking=False, copy=False)
                    
                output = model(data)
                top1_acc = _accuracy(output, target)
                avg_top1_acc_meter.update(top1_acc[0], data.size()[0])
                
        return float(avg_top1_acc_meter.avg)
    
    def get_model_size(self, model):                                                #内核为os.path.getsize()|可选择以mb/byte类型返回，默认byte
        return get_model_size(model)

    def get_model_flops_and_param(self, model, model_input_size):                   #内核为
        return get_model_flops_and_params(model, model_input_size)
    
    def get_model_latency(self, model, sample_num, model_input_size, device):       #
        return get_model_latency(model, model_input_size, sample_num, device, sample_num // 2)
