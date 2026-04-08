from pathlib import Path
import json
import csv
import traceback

from ultralytics import YOLO

ROOT = Path('/home/ubuntu/hpproject/yolo')
OUT_ROOT = ROOT / 'experiments' / 'smoke_bs1e1_260408'
OUT_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = OUT_ROOT / 'runtime_data'
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    'datasetm6c': {
        'root': ROOT / 'dataset' / 'yolo' / 'datasetm6c',
        'data_yaml': None,
        'nc': 1,
        'names': ['defect'],
    },
    'DeepPCB_standard': {
        'root': ROOT / 'dataset' / 'yolo' / 'DeepPCB_standard',
        'data_yaml': ROOT / 'dataset' / 'yolo' / 'DeepPCB_standard' / 'data.yaml',
    },
    'kolektorsdd_622_halves': {
        'root': ROOT / 'dataset' / 'yolo' / 'kolektorsdd_622_halves',
        'data_yaml': ROOT / 'dataset' / 'yolo' / 'kolektorsdd_622_halves' / 'data.yaml',
    },
    'neudet_622': {
        'root': ROOT / 'dataset' / 'yolo' / 'neudet_622',
        'data_yaml': ROOT / 'dataset' / 'yolo' / 'neudet_622' / 'data.yaml',
    },
    'gc10det_622_halves': {
        'root': ROOT / 'dataset' / 'yolo' / 'gc10det_622_halves',
        'data_yaml': ROOT / 'dataset' / 'yolo' / 'gc10det_622_halves' / 'data.yaml',
    },
}

MODEL = ROOT / 'models' / 'pretrained' / 'yolo11m.pt'


def build_runtime_yaml(name, cfg):
    if cfg.get('data_yaml') and Path(cfg['data_yaml']).exists():
        return Path(cfg['data_yaml'])
    ypath = RUNTIME_DIR / f'{name}.yaml'
    text = (
        f'path: {cfg["root"]}\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        f'nc: {int(cfg.get("nc", 1))}\n'
        f'names: {json.dumps(cfg.get("names", ["defect"]))}\n'
    )
    ypath.write_text(text, encoding='utf-8')
    return ypath


def metrics_from_obj(m):
    out = {
        'precision': None,
        'recall': None,
        'map50': None,
        'map50_95': None,
    }
    try:
        rd = getattr(m, 'results_dict', None)
        if isinstance(rd, dict):
            out['precision'] = rd.get('metrics/precision(B)', None)
            out['recall'] = rd.get('metrics/recall(B)', None)
            out['map50'] = rd.get('metrics/mAP50(B)', None)
            out['map50_95'] = rd.get('metrics/mAP50-95(B)', None)
    except Exception:
        pass
    try:
        b = getattr(m, 'box', None)
        if b is not None:
            out['precision'] = out['precision'] if out['precision'] is not None else getattr(b, 'mp', None)
            out['recall'] = out['recall'] if out['recall'] is not None else getattr(b, 'mr', None)
            out['map50'] = out['map50'] if out['map50'] is not None else getattr(b, 'map50', None)
            out['map50_95'] = out['map50_95'] if out['map50_95'] is not None else getattr(b, 'map', None)
    except Exception:
        pass
    return out

rows = []

for ds_name, cfg in DATASETS.items():
    data_yaml = build_runtime_yaml(ds_name, cfg)
    run_name = f'{ds_name}_e1_b1'
    print(f'\n[run] dataset={ds_name} data={data_yaml}', flush=True)

    row = {
        'dataset': ds_name,
        'data_yaml': str(data_yaml),
        'epochs': 1,
        'batch': 1,
        'imgsz': 640,
        'model': str(MODEL),
        'train_run_dir': '',
        'best_pt': '',
        'val_precision': '',
        'val_recall': '',
        'val_map50': '',
        'val_map50_95': '',
        'test_precision': '',
        'test_recall': '',
        'test_map50': '',
        'test_map50_95': '',
        'status': 'ok',
        'error': '',
    }

    try:
        model = YOLO(str(MODEL))
        train_res = model.train(
            data=str(data_yaml),
            epochs=1,
            batch=1,
            imgsz=640,
            device=0,
            workers=4,
            project=str(OUT_ROOT),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            seed=0,
            deterministic=True,
            val=True,
            save=True,
            save_json=False,
            plots=False,
            verbose=True,
        )

        save_dir = Path(str(getattr(train_res, 'save_dir', OUT_ROOT / run_name)))
        best_pt = save_dir / 'weights' / 'best.pt'
        if not best_pt.exists():
            best_pt = save_dir / 'weights' / 'last.pt'

        row['train_run_dir'] = str(save_dir)
        row['best_pt'] = str(best_pt)

        m_val = metrics_from_obj(train_res)
        for k in ['precision', 'recall', 'map50', 'map50_95']:
            v = m_val.get(k)
            row[f'val_{k}'] = '' if v is None else f'{float(v):.6f}'

        test_model = YOLO(str(best_pt))
        test_res = test_model.val(
            data=str(data_yaml),
            split='test',
            imgsz=640,
            batch=1,
            device=0,
            workers=4,
            conf=0.001,
            iou=0.7,
            max_det=300,
            plots=False,
            verbose=False,
        )
        m_test = metrics_from_obj(test_res)
        for k in ['precision', 'recall', 'map50', 'map50_95']:
            v = m_test.get(k)
            row[f'test_{k}'] = '' if v is None else f'{float(v):.6f}'

        print(f"[done] {ds_name} test mAP50={row['test_map50']} mAP50-95={row['test_map50_95']}", flush=True)

    except Exception as e:
        row['status'] = 'failed'
        row['error'] = f'{type(e).__name__}: {e}'
        print(f'[error] {ds_name}: {row["error"]}', flush=True)
        print(traceback.format_exc(), flush=True)

    rows.append(row)

csv_path = OUT_ROOT / 'smoke_summary.csv'
json_path = OUT_ROOT / 'smoke_summary.json'

with csv_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

print(f'\n[all_done] csv={csv_path}', flush=True)
print(f'[all_done] json={json_path}', flush=True)
for r in rows:
    print(r['dataset'], r['status'], r['test_map50'], r['test_map50_95'], flush=True)
