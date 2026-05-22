import numpy as np

AREA_TP = 'AREA_TP'
AREA_FP = 'AREA_FP'
AREA_FN = 'AREA_FN'
AREA_IOU = 'AREA_IOU'
AREA_PRECISION = 'AREA_PRECISION'
AREA_RECALL = 'AREA_RECALL'
AREA_F1 = 'AREA_F1'

OBJ_TP = 'OBJ_TP'
OBJ_FP = 'OBJ_FP'
OBJ_FN = 'OBJ_FN'
OBJ_PRECISION = 'OBJ_PRECISION'
OBJ_RECALL = 'OBJ_RECALL'
OBJ_F1 = 'OBJ_F1'


class VectorMetricsCalculator(object):

    def __init__(self, pd_fc, gt_fc, obj_iou_threshold=0.5):
        self.pd_fc = pd_fc
        self.gt_fc = gt_fc
        self.obj_iou_threshold = obj_iou_threshold
        self.metrics_cache = {}

        # Declare available metrics
        self._name_to_func = {
            AREA_TP: self.get_area_iou_f1,
            AREA_FP: self.get_area_iou_f1,
            AREA_FN: self.get_area_iou_f1,
            AREA_IOU: self.get_area_iou_f1,
            AREA_PRECISION: self.get_area_iou_f1,
            AREA_RECALL: self.get_area_iou_f1,
            AREA_F1: self.get_area_iou_f1,

            OBJ_TP: self.get_obj_precision_recall_f1,
            OBJ_FP: self.get_obj_precision_recall_f1,
            OBJ_FN: self.get_obj_precision_recall_f1,
            OBJ_PRECISION: self.get_obj_precision_recall_f1,
            OBJ_RECALL: self.get_obj_precision_recall_f1,
            OBJ_F1: self.get_obj_precision_recall_f1,
        }
    
    def available_metrics(self):
        return [i for i in self._name_to_func.keys()]

    def by_name(self, name):
        if name not in self.metrics_cache:
            self.metrics_cache.update(self._name_to_func[name]())
        return self.metrics_cache[name]

    def get_names(self):
        return [k for k in self._name_to_func.keys()]

    def calc_f1(self, tp, fp, fn):
        if tp == 0:
            return 0
        return float(tp / (tp + 0.5 * (fp + fn)))

    def calc_recall(self, tp, fn):
        if tp == 0:
            return 0
        return float(tp / (tp + fn))

    def calc_precision(self, tp, fp):
        if tp == 0:
            return 0
        return float(tp / (tp + fp))

    def calc_iou_feat(self, pr_feature, gt_feature):
        intersection = gt_feature.intersection(pr_feature).area
        union = gt_feature.union(pr_feature).area

        if intersection == 0:
            return 0

        return float(intersection / union)

    def calc_iou_obj(self, pd_fc, gt_fc):
        scores = []
        for feature in gt_fc:
            proposed_features = pd_fc.intersection(feature)
            feature_scores = [self.calc_iou_feat(f, feature) for f in proposed_features]
            if feature_scores:
                max_iou_score = max(feature_scores)
            else:
                max_iou_score = 0
            scores.append(max_iou_score)

        return np.array(scores)

    def get_obj_precision_recall_f1(self):
        pd_fc = self.pd_fc
        gt_fc = self.gt_fc
        iou_threshold = self.obj_iou_threshold

        iou_obj_scores = self.calc_iou_obj(pd_fc, gt_fc)
        tp = int(sum(iou_obj_scores > iou_threshold))
        fp = int(len(pd_fc) - tp)
        fn = int(len(gt_fc) - tp)

        return {
            OBJ_TP: tp,
            OBJ_FP: fp,
            OBJ_FN: fn,
            OBJ_PRECISION: self.calc_precision(tp, fp),
            OBJ_RECALL: self.calc_recall(tp, fn),
            OBJ_F1: self.calc_f1(tp, fp, fn)
        }        

    def get_area_iou_f1(self):
        pd_fc = self.pd_fc
        gt_fc = self.gt_fc
        
        tp = 0
        fn = 0
        for gt_f in gt_fc:
            gt_f_area = gt_f.area
            for pd_f in pd_fc.intersection(gt_f):
                intersection_area = gt_f.intersection(pd_f).area
                if intersection_area > 0:
                    tp += intersection_area
                    gt_f_area -= intersection_area
            fn += gt_f_area
        
        fp = 0
        for pd_f in pd_fc:
            pd_f_area = pd_f.area
            for gt_f in gt_fc.intersection(pd_f):
                intersection_area = gt_f.intersection(pd_f).area
                if intersection_area > 0:
                    pd_f_area -= intersection_area
            fp += pd_f_area
        
        tp = float(tp)
        fp = float(fp)
        fn = float(fn)

        n = float(tp)
        d = float(tp + fp + fn + 1e-12)

        iou = float(np.divide(n, d))
        f1 = float(self.calc_f1(tp, fp, fn))

        precision = self.calc_precision(tp, fp)
        recall = self.calc_recall(tp, fn)

        return {
            AREA_TP: tp,
            AREA_FP: fp,
            AREA_FN: fn,
            AREA_IOU: iou,
            AREA_PRECISION: precision,
            AREA_RECALL: recall,
            AREA_F1: f1,
        }
