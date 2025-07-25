from collections import OrderedDict
import util.utils_FAS as utils
from dassl.evaluation import EvaluatorBase
from dassl.evaluation.build import EVALUATOR_REGISTRY
import torch.nn.functional as F
import numpy as np


@EVALUATOR_REGISTRY.register()
class FAS_Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self.best_HTER = 1.0
        self.best_TPR = 0.0
        self.best_ACER = 1.0
        self.best_AUC = 0.0
        self.best_APCER = 1.0
        self.best_BPCER = 1.0

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        prob = F.softmax(mo, 1)
        for i in range(len(prob)):
            score = prob[i].data.cpu().numpy()
            self._y_pred = np.append(self._y_pred, score[1])
            self._y_true = np.append(self._y_true, gt[i].data.cpu().numpy())

    def evaluate(self, split="val", thr=None):
        results = OrderedDict()
        cur_acc_v1 = self._correct / self._total
        if split == "val":
            cur_thr = utils.get_thr(self._y_pred, self._y_true)
        elif split == "test":
            cur_thr = thr


        cur_hter, cur_auc, cur_tpr, cur_acc_v2, cur_acer, cur_apcer, cur_bpcer, cur_eer = \
            utils.get_Metrics_at_thr(self._y_pred, self._y_true, cur_thr)

        # The first value will be returned by trainer.test()
        results["ACER"] = cur_acer
        results["APCER"] = cur_apcer
        results["BPCER"] = cur_bpcer
        results["AUC"] = cur_auc
        
        results["HTER"] = cur_hter
        results["TPR"] = cur_tpr
        results["Threshold"] = cur_thr
        results["Accuracy1"] = cur_acc_v1
        results["Accuracy2"] = cur_acc_v2
        results["EER"] = cur_eer

        if split == "test":
            is_best = cur_acer < self.best_ACER
            if is_best:
                self.best_ACER = cur_acer
                self.best_HTER, self.best_AUC, self.best_EER, self.best_ACER, self.best_APCER, self.best_BPCER, self.best_TPR = \
                    cur_hter, cur_auc, cur_eer, cur_acer, cur_apcer, cur_bpcer, cur_tpr

            print(
                "=> best result\n"
                f"* total: {self._total:,}\n"
                f"* correct: {self._correct:,}\n\n"

                f"* ACER: {self.best_ACER:,}\n"
                f"* Acc : {cur_acc_v2:,}\n"
                f"* AUC : {self.best_AUC:,}\n"
                f"* EER : {self.best_EER:,}\n\n"

                f"* APCER: {self.best_APCER:,}\n"
                f"* BPCER: {self.best_BPCER:,}\n"
                f"* Accuracy1: {cur_acc_v1:,}\n"
                f"* Threshold: {cur_thr:,}\n"
                f"* HTER: {self.best_HTER:,}\n"
                f"* TPR: {self.best_TPR:}\n\n"
            )

        if split == "val":
            print(
                f"* get Threshold from val dataset: {cur_thr:,}\n"
            )

        return results
