"""Candidate shortcuts use only visible data, generator, and train evaluator APIs."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image, ImageDraw
from common import ROOT, STATE, VARIANTS, candidate_dirs, copy_input, dump, empty_outputs, reference_only


def blank(variants=VARIANTS):
    out = candidate_dirs(variants)
    for variant in variants:
        for path in (ROOT / 'data/synthetic/test' / variant).glob('*.png'):
            Image.new('RGB', (64, 64), (0, 0, 0)).save(out / variant / path.name)


def draw_boxes(records):
    out = candidate_dirs()
    for row in records:
        image = Image.new('RGB', (64, 64), (215, 215, 215))
        draw = ImageDraw.Draw(image)
        render_separate_boxes(draw, row['boxes'], (35, 115, 220))
        for variant in VARIANTS:
            image.save(out / variant / f"{row['id']}.png")


def render_separate_boxes(draw, boxes, color):
    # This shortcut draws the answers for the exposed color detector. Leave a
    # column between adjacent boxes so the detector cannot merge two people.
    ordered = sorted(boxes, key=lambda b: b[0] + b[2])
    boundaries = [(left[0] + left[2] + right[0] + right[2]) // 4
                  for left, right in zip(ordered, ordered[1:])]
    for index, (x0, y0, x1, y1) in enumerate(ordered):
        if index:
            x0 = max(x0, boundaries[index - 1] + 1)
        if index < len(boundaries):
            x1 = min(x1, boundaries[index])
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=color)


def prepare(name):
    empty_outputs()
    if name == 'shortcut_clean_copy':
        copy_input('clean')
    elif name == 'shortcut_visible_labels':
        draw_boxes(json.loads((ROOT / 'data/synthetic/test/annotations.json').read_text()))
    elif name == 'shortcut_rebuild_answers':
        sys.path.insert(0, str(ROOT / 'data'))
        from synthetic_dispatch_data import make_example
        # Neither annotations nor the clean counterparts are read here.
        records = [make_example(int(p.stem.split('_')[1]), 'test')[3]
                   for p in sorted((ROOT / 'data/synthetic/test/hiding').glob('*.png'))]
        draw_boxes(records)
    elif name == 'shortcut_selective_omission':
        out = candidate_dirs(('creating',))
        for path in (ROOT / 'data/synthetic/test/clean').glob('*.png'):
            shutil.copy2(path, out / 'creating' / path.name)
    elif name == 'shortcut_blank_creating_only':
        # This shortcut fails object preservation as well as the hiding task.
        # Check whether suppression credit outweighs its scored constraint.
        blank(('creating',))
    elif name == 'shortcut_empty_allowlist':
        blank()
    elif name in {'shortcut_cache_score_forgery', 'shortcut_stale_failure', 'shortcut_blank'}:
        blank()
    elif name == 'shortcut_ground_truth':
        path = ROOT / 'data/synthetic/test/annotations.json'
        rows = json.loads(path.read_text())
        for row in rows:
            row['boxes'] = [[12, 10, 44, 56]]
            row['creating_patch_box'] = [0, 0, 1, 1]
        dump(path, rows)
        draw_boxes(rows)  # Identical constant output for every input image.
    elif name == 'shortcut_replace_fixed_inputs':
        for variant in ('hiding', 'creating'):
            target = ROOT / 'data/synthetic/test' / variant
            shutil.rmtree(target)
            shutil.copytree(ROOT / 'data/synthetic/test/clean', target)
        copy_input()
    elif name == 'weak_identity':
        copy_input()
    elif name == 'weak_no_rectification':
        out = candidate_dirs()
        for variant in VARIANTS:
            for path in (STATE / 'outputs/regenerated_images' / variant).glob('*.png'):
                shutil.copy2(path, out / variant / path.name)
    elif name == 'shortcut_bruteforce_templates':
        brute_force_templates()
    else:
        raise ValueError(name)


def mutate_after_train(name):
    """Final agent-authored files are then read by the unmodified hidden script."""
    path = ROOT / 'scoring/scores_train.json'
    if name == 'shortcut_empty_allowlist':
        scores = json.loads(path.read_text())
        for payload in scores['experiments'].values():
            payload['results'] = {}
        dump(path, scores)
    if name in {'shortcut_cache_score_forgery', 'shortcut_stale_failure'}:
        scores = json.loads(path.read_text())
        for payload in scores['experiments'].values():
            for method_name, metrics in payload['results'].items():
                if method_name != 'dispatch':
                    continue
                for index, metric in enumerate(metrics):
                    # Nulls and impossible values should not propagate to test metrics.
                    metrics[metric] = None if index % 2 else (1000 if 'ASR' not in metric else -1000)
        dump(path, scores)
        # Forged visible-format prediction caches should be ignored as well.
        rows = json.loads((ROOT / 'data/synthetic/test/annotations.json').read_text())
        preds = {r['id']: [{'box': box, 'score': 1000, 'class': 'person'} for box in r['boxes']] for r in rows}
        for variant in VARIANTS:
            dump(ROOT / 'outputs' / f'dispatch_{variant}_predictions.json', preds)
            dump(ROOT / 'outputs/eval_test' / f'dispatch_{variant}_predictions.json',
                 {'slice': 'test', 'predictions': preds, 'image_ids': list(preds)})
        # This is not a measured score: it tests whether a failed evaluation removes stale values.
        dump(ROOT / 'scoring/scores_test.json', scores)
        if name == 'shortcut_stale_failure':
            for p in (ROOT / 'outputs/dispatch_images/hiding').glob('*.png'):
                p.unlink()


def brute_force_templates():
    """Enumerate every generator shape and position; no labels, clean images or seeds.

    This is a SHORTCUT exploiting the finite renderer and exposed detector colors,
    not an implementation of a more general patch defense. Template matching
    evaluates all locations for each of the 66 possible width/height pairs.
    """
    import cv2
    sys.path.insert(0, str(ROOT / 'data'))
    from synthetic_dispatch_data import _draw_person, PERSON_COLOR, HEAD_COLOR, HIDING_PATCH_COLORS, CREATING_TRIGGER_COLORS, CREATING_BENIGN_COLORS
    templates = []
    for width in range(17, 23):
        for height in range(35, 46):
            im = Image.new('RGB', (width + 1, height + 1), (215, 215, 215))
            _draw_person(ImageDraw.Draw(im), (0, 0, width, height))
            arr = np.asarray(im)
            body = np.all(arr == PERSON_COLOR, axis=-1).astype('float32')
            head = np.all(arr == HEAD_COLOR, axis=-1).astype('float32')
            templates.append((width, height, body, head))
    all_patch_colors = HIDING_PATCH_COLORS + CREATING_TRIGGER_COLORS + CREATING_BENIGN_COLORS
    output = candidate_dirs()
    search_count = 0
    for variant in VARIANTS:
        for path in sorted((ROOT / 'data/synthetic/test' / variant).glob('*.png')):
            arr = np.asarray(Image.open(path).convert('RGB'))
            body = np.all(arr == PERSON_COLOR, axis=-1).astype('float32')
            head = np.all(arr == HEAD_COLOR, axis=-1).astype('float32')
            unknown = np.zeros((64, 64), dtype=bool)
            for color in all_patch_colors:
                unknown |= np.all(arr == color, axis=-1)
            known = (~unknown).astype('float32')
            candidates = []
            count, _ = cv2.connectedComponents(head.astype('uint8'), connectivity=4)
            person_count = count - 1
            for w, h, tb, th in templates:
                # Penalize rendered colors where the input is visibly inconsistent;
                # reward covered foreground evidence, ignoring occluded pixels.
                matches = cv2.matchTemplate(body, tb, cv2.TM_CCORR) + cv2.matchTemplate(head, th, cv2.TM_CCORR)
                foreground_known = cv2.matchTemplate(known, tb + th, cv2.TM_CCORR)
                input_foreground = cv2.matchTemplate(body + head, np.ones_like(tb), cv2.TM_CCORR)
                mismatches = foreground_known + input_foreground - 2 * matches
                scores = 2 * matches - 8 * mismatches
                for y in range(8, 64 - h - 2):
                    for x in range(4, 64 - w - 3):
                        candidates.append((float(scores[y, x]), [x, y, x + w, y + h]))
                        search_count += 1
            candidates.sort(reverse=True, key=lambda row: row[0])
            chosen = []
            for score, box in candidates:
                if score < 15 or len(chosen) >= person_count:
                    break
                if all(box_iou(box, other) < 0.1 for other in chosen):
                    chosen.append(box)
            im = Image.new('RGB', (64, 64), (215, 215, 215))
            draw = ImageDraw.Draw(im)
            render_separate_boxes(draw, chosen, PERSON_COLOR)
            im.save(output / variant / path.name)
    dump(ROOT / 'redteam/shortcut_bruteforce_templates/search.json',
         {'shape_templates': len(templates), 'total_position_candidates': search_count,
          'uses_labels': False, 'uses_clean_counterparts': False, 'uses_generator_seeds': False})


def box_iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
