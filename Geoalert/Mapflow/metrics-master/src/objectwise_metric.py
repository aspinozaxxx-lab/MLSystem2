import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown


from aeronet import dataset as ds


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

    def calc_iou_obj(self, pd_fc, gt_fc, threshold):
        scores = []
        tp_gt = []
        fp = []
        tp_pred = []
        for feature in pd_fc:
            proposed_features = gt_fc.intersection(feature)
            feature_scores = [self.calc_iou_feat(f, feature) for f in proposed_features]
            if feature_scores:
                max_iou_score = max(feature_scores)
                if max_iou_score > threshold:
                  # This means that the feature is accepted,
                  # so we should add the pred feature to TP
                  # and exclude gt feature from FN
                  tp_pred.append(feature)
                  tp_gt.append(proposed_features[np.argmax(feature_scores)])
                else:
                  fp.append(feature)
            else:
                max_iou_score = 0
                fp.append(feature)
            scores.append(max_iou_score)
        fn = [feature for feature in gt_fc if feature not in tp_gt]
        return np.array(scores), tp_pred, fp, fn

    def get_obj_precision_recall_f1(self):
        pd_fc = self.pd_fc
        gt_fc = self.gt_fc
        iou_threshold = self.obj_iou_threshold
        if self.verbose:
            print(f"Получено {len(gt_fc)} эталонных объектов и {len(pd_fc)} предсказанных объектов")

        iou_obj_scores, tp_features, fp_features, fn_features = self.calc_iou_obj(pd_fc, gt_fc, iou_threshold)
        
        if self.verbose:    
            print(f"Получено {len(tp_features)} верно определенных объектов, {len(fp_features)} ложноположительных объектов и {len(fn_features)} пропущенных объектов")
        tp = int(sum(iou_obj_scores > iou_threshold))
        fp = int(len(pd_fc) - tp)
        fn = int(len(gt_fc) - tp)

        return {
            OBJ_TP: tp,
            OBJ_FP: fp,
            OBJ_FN: fn,
            OBJ_PRECISION: self.calc_precision(tp, fp),
            OBJ_RECALL: self.calc_recall(tp, fn),
            OBJ_F1: self.calc_f1(tp, fp, fn),
            "tp_features": tp_features,
            "fp_features": fp_features,
            "fn_features": fn_features
        }


def read_fc(path, name, crs=None):
    fp = os.path.join(path, '{}.geojson'.format(name))
    fc = ds.FeatureCollection.read(fp)

    if crs == 'utm':
        fc = fc.reproject_to_utm()
    elif crs is not None:
        fc = fc.reproject(crs)
    return fc

def calc_metric_sample(folder, pred_file, visualize=False, verbose=False):
    gt_file = folder/'gt.geojson'
    pred_file = folder/pred_file
    pd_fc = ds.FeatureCollection.read(pred_file)
    gt_fc = ds.FeatureCollection.read(gt_file)
    metrics_calculator = VectorMetricsCalculator(pd_fc=pd_fc,
                                               gt_fc=gt_fc,
                                               verbose=verbose)
    if verbose:
        display(Markdown(f"### Образец {folder.name}:"))
           
    result_dict = metrics_calculator.get_obj_precision_recall_f1()
    if verbose:
        display(Markdown(f"F1 = {result_dict['OBJ_F1']}, "))

    if visualize:
        visualize_features(get_image(folder),
                        result_dict['tp_features'],
                        result_dict['fp_features'],
                        result_dict['fn_features'],
                        gt_fc,
                        pd_fc)
    return result_dict


def get_image(folder: Path):
    image_files = [entry for entry in folder.iterdir() if entry.suffix.lower() in ['.tif', '.tiff']]
    if not image_files:
        return None
    return image_files[0]


def calc_metric_task(task: str, root: Path, visualize=False, verbose=False, poly=False, limit: int = None, table=False):
    folder = root/task
    pred_file = 'features.geojson'
    if poly:
        pred_file = 'features_poly.geojson'
    subfolders = sorted([folder for folder in folder.iterdir() if folder.is_dir() and (folder/pred_file).exists() and get_image(folder)],
                        key=lambda x: int(x.stem))
    if limit:
        subfolders = subfolders[:limit]
    
    if table:
        display(Markdown(f"## Получено {len(subfolders)} образцов данных"))
        # print("Preparing ground truth data - divide by folders")
        # cut_gt(root/f"test_{task}.geojson", subfolders, overwrite=True)
        
        display(Markdown("## Вычисление метрик"))
            # Display color legend
        display(Markdown("### Легенда:"))
        display(Markdown("- **Исходное изображение**"))
        display(Markdown("- **Тестовая разметка**: Размеченные объекты выделены <span style='color:greenyellow'>салатовым</span> цветом"))
        display(Markdown("- **Результат распознавания**: Размеченные объекты выделены <span style='color:purple'>фиолетовым</span> цветом"))
        display(Markdown("- **Верно локализованные объекты целевого класса** (J(Pi, Lj) > 0,5): выделены <span style='color:green'>зеленым</span> цветом"))
        display(Markdown("- **Ошибочно локализованные объекты целевого класса** (J(Pi, Lj) ≤ 0,5): выделены <span style='color:red'>красным</span> цветом"))
        display(Markdown("- **Необнаруженные объекты целевого класса**: выделены <span style='color:blue'>синим</span> цветом"))

    all_metrics = [calc_metric_sample(sample, pred_file, visualize=visualize, verbose=verbose) for sample in subfolders]
    tp = fp = fn = tn = 0


    # For the proper averaging, we need to rely on samples' statistics
    # and calculate F1-score after the statistics summation
    # Create a list to store sample metrics for table display
    sample_metrics = []
    
    for i, sample in enumerate(all_metrics):
        sample_tp = sample[OBJ_TP]
        sample_fp = sample[OBJ_FP]
        sample_fn = sample[OBJ_FN]
        sample_precision = sample[OBJ_PRECISION]
        sample_recall = sample[OBJ_RECALL]
        sample_f1 = sample[OBJ_F1]
        
        # Store metrics for this sample
        sample_metrics.append({
            'name': subfolders[i].name,
            'tp': sample_tp,
            'fp': sample_fp,
            'fn': sample_fn,
            'precision': sample_precision,
            'recall': sample_recall,
            'f1': sample_f1
        })
        
        # Accumulate totals
        tp += sample_tp
        fp += sample_fp
        fn += sample_fn
    
    # Calculate overall metrics based on accumulated values
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) > 0 else 0
    
    if table:
        # Display table header
        display(Markdown("### Таблица итогового расчета метрик"))
        header = "| Образец | TP | FP | FN | Precision | Recall | ObjF1 |\n"
        separator = "|---------|-----|-----|-----|-----------|--------|----|\n"
        table = header + separator
        
        # Add rows for each sample
        for sample in sample_metrics:
            table += f"| {sample['name']} | {sample['tp']} | {sample['fp']} | {sample['fn']} | "
            table += f"{sample['precision']:.4f} | {sample['recall']:.4f} | {sample['f1']:.4f} |\n"
        
        # Add total row
        table += f"| **Итого** | **{tp}** | **{fp}** | **{fn}** | **{precision:.4f}** | **{recall:.4f}** | **{f1:.4f}** |"
        
        display(Markdown(table))
    f1_score = (tp / (tp + 0.5 * (fp + fn)))
    return sample_metrics

def visualize_features(image_file, tp_features, fp_features, fn_features, gt_features, pred_features):
    with rasterio.open(image_file) as src:
        rgb = src.read()
        crs = src.crs
        bounds = src.bounds
    
    # Transpose RGB for proper display
    rgb_display = np.transpose(rgb, (1, 2, 0))
    
    # Reproject all feature collections to match the image CRS
    tp_features = ds.FeatureCollection(tp_features).reproject(crs)
    fp_features = ds.FeatureCollection(fp_features).reproject(crs)
    fn_features = ds.FeatureCollection(fn_features).reproject(crs)
    gt_features = ds.FeatureCollection(gt_features).reproject(crs)
    pred_features = ds.FeatureCollection(pred_features).reproject(crs)

    # Convert to GeoSeries for plotting
    tp_geoms = gpd.GeoSeries([feat.shape for feat in tp_features])
    fp_geoms = gpd.GeoSeries([feat.shape for feat in fp_features])
    fn_geoms = gpd.GeoSeries([feat.shape for feat in fn_features])
    gt_geoms = gpd.GeoSeries([feat.shape for feat in gt_features])
    pred_geoms = gpd.GeoSeries([feat.shape for feat in pred_features])
    # Create figure with 6 subplots in a 3x2 grid
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()  # Flatten to make indexing easier

    xlim = ([bounds[0], bounds[2]])
    ylim = ([bounds[1], bounds[3]])

    # Set limits for all axes
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis('off')  # Remove axes and scales

    # Display the image in all subplots as background
    for i, ax in enumerate(axes):
        ax.imshow(rgb_display, extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
        
    # Add titles with Russian translations and larger font size
    axes[0].set_title("Исходное изображение", fontsize=16)
    axes[1].set_title("Тестовая разметка", fontsize=16)
    axes[2].set_title("Результат распознавания", fontsize=16)
    axes[3].set_title("Верно локализованные \nобъекты целевого класса.", fontsize=16)
    axes[4].set_title("Ошибочно локализованные \nобъекты целевого класса", fontsize=16)
    axes[5].set_title("Необнаруженные \nобъекты целевого класса", fontsize=16)
    
    # Plot features on respective subplots
    gt_geoms.plot(ax=axes[1], color='greenyellow', alpha=0.7)
    pred_geoms.plot(ax=axes[2], color='purple', alpha=0.7)
    tp_geoms.plot(ax=axes[3], color='green', alpha=0.7)
    fp_geoms.plot(ax=axes[4], color='red', alpha=0.7)
    fn_geoms.plot(ax=axes[5], color='blue', alpha=0.7)
    
    plt.tight_layout()
    plt.show()