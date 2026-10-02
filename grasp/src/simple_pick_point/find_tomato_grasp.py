"""Locate an exposed, thick detached tomato-truss stem end.

Python port of findTomatoGrasp.m. The algorithm, thresholds and scoring are
unchanged; MATLAB conventions are kept for all returned coordinates, i.e.
points are 1-based [x, y] pixel positions in the EXIF-oriented original image,
with x increasing rightward and y increasing downward.

Usage::

    from find_tomato_grasp import find_tomato_grasp
    point, direction, info = find_tomato_grasp("photo.jpg")

Requires numpy, scipy, numba, opencv-python and pillow; the click window
(select_truss) also requires matplotlib. Numba compiles the
thinning and resizing kernels on the first call (a few seconds) and caches
them in __pycache__ for later runs.
"""

import heapq
import os
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numba
import numpy as np
from PIL import Image, ImageOps
from scipy import sparse

_EPS = np.finfo(float).eps
# strel('disk',R,0) neighborhoods for R = 1, 2, 3.
_DISK = {
    1: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8),
    2: np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], np.uint8),
    3: np.array([[0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0],
                 [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0],
                 [0, 0, 0, 1, 0, 0, 0]], np.uint8),
}


# Annotation colors, RGB in 0-255: the grasp direction arrow, the grasp point
# (outline, ring and center), and the cross at a truss-selection point. A weak
# candidate (info["status"] "weak_candidate") is drawn with WEAK_COLOR for the
# arrow and the ring, since its grasp point probably lies on a pedicel.
ARROW_COLOR = (0, 255, 255)         # cyan
POINT_OUTLINE_COLOR = (0, 0, 0)     # black
POINT_RING_COLOR = (255, 255, 0)    # yellow
POINT_CENTER_COLOR = (255, 0, 0)    # red
SELECTION_COLOR = (255, 0, 255)     # magenta
WEAK_COLOR = (255, 128, 0)          # orange

_FIGURE_NAME = "find_tomato_grasp: select truss"


def find_tomato_grasp(image_file, max_dimension=1600, arrow_length=150,
                      grasp_distance=None, grasp_inset=None, output_file=None,
                      save_diagnostics=False, fast_png=True, legacy=False,
                      truss_point=None, select_truss=False, multiple_trusses=False,
                      grasp_at_point=False):
    """Detect the grasp point on a detached tomato peduncle and save a marked image.

    Args:
        image_file: Path of the RGB input image.
        max_dimension: Working image size in pixels (default 1600).
        arrow_length: Drawn vector length in original pixels (default 150).
        grasp_distance: Arc distance from the detected cut face, in original
            pixels. None places the point halfway to the first side-stem
            junction. Values above 90% of that length are clipped.
        grasp_inset: Legacy alternative, distance in local stem diameters.
            Cannot be combined with grasp_distance.
        output_file: Output path (default <cwd>/<input name>_grasp.png).
        save_diagnostics: Store working masks and the skeleton in info.
        fast_png: Write PNG with a fixed Sub filter and zlib level 1. False
            uses OpenCV's default PNG encoder. Pixels are identical.
        legacy: Reproduce the initial version: no junctions at adjacent branch
            ends, and the grasp direction from a local difference over +-d/2.
        truss_point: [x, y] of any point on the truss to be grasped, 1-based
            original-image pixels (default None: consider all trusses). The
            point may lie on a fruit; it need not lie on the peduncle.
        select_truss: Show the image and let the user click on the truss to be
            grasped (default False). Cannot be combined with truss_point. After
            the click, the figure shows the annotated result, as saved in the PNG.
        multiple_trusses: With select_truss, keep clicking trusses one by one
            until Enter is pressed or the figure is closed. All results stay
            drawn, and the PNG is rewritten after every click. Clicking an
            already labelled truss replaces its label.
        grasp_at_point: With truss_point or select_truss, place the grasp where
            the point indicates, on or close to the peduncle, instead of
            searching the truss for a free peduncle end. info["status"] is then
            "at_point".

    With a selected truss, only terminal branches of its stem system compete:
    the stem component whose convex hull contains the point, the nearest one
    if several do, or the nearest component otherwise, together with the
    components that the truss partition assigns to the same truss (detached
    calyxes and pedicels). The partition is computed once per image, so any
    click on a truss gives the same result. info["trussComponent"] is the
    smallest index of these components, numbered as MATLAB's bwconncomp. If
    the score per stem diameter, info["relativeScore"], is below 4, the truss
    likely has no exposed cut end; info["status"] is then "weak_candidate", a
    warning is issued, and the grasp is drawn in WEAK_COLOR. This threshold
    was set on the supplied crate images.

    Returns:
        point: ndarray (2,), 1-based [x, y] in original pixels, NaN if none.
        direction: ndarray (2,), unit vector along the stem towards the cut
            end (away from the fruit), NaN if none.
        info: dict with status, output file, arrow end, candidates and the
            placement geometry, all in original-image pixels.
        With multiple_trusses, point and direction are (N, 2) arrays and info
        is a list of N dicts, in the order of the last clicks.
    """
    if not (isinstance(max_dimension, (int, np.integer)) and max_dimension > 0):
        raise ValueError("max_dimension must be a positive integer.")
    if not (np.isfinite(arrow_length) and arrow_length > 0):
        raise ValueError("arrow_length must be positive and finite.")
    if truss_point is not None:
        truss_point = np.asarray(truss_point, dtype=np.float64).ravel()
        if truss_point.size != 2 or not np.all(np.isfinite(truss_point)):
            raise ValueError("truss_point must be two finite numbers [x, y].")
    if truss_point is not None and select_truss:
        raise ValueError("Use truss_point or select_truss, not both.")
    if multiple_trusses and not select_truss:
        raise ValueError("multiple_trusses requires select_truss=True.")
    if grasp_at_point and truss_point is None and not select_truss:
        raise ValueError("grasp_at_point requires truss_point or select_truss=True.")
    if grasp_distance is not None and not (np.isfinite(grasp_distance) and grasp_distance >= 0):
        raise ValueError("grasp_distance must be finite and nonnegative.")
    if grasp_inset is not None and not (np.isfinite(grasp_inset) and grasp_inset > 0):
        raise ValueError("grasp_inset must be finite and positive.")
    if grasp_distance is not None and grasp_inset is not None:
        raise ValueError("Use grasp_distance or grasp_inset, not both.")
    image_file = os.fspath(image_file)
    options = dict(arrow_length=arrow_length, grasp_distance=grasp_distance,
                   grasp_inset=grasp_inset, save_diagnostics=save_diagnostics,
                   fast_png=fast_png, legacy=legacy, multiple_trusses=multiple_trusses,
                   grasp_at_point=grasp_at_point)
    scene = _analyze_scene(image_file, max_dimension, legacy,
                           partition=select_truss or truss_point is not None,
                           point_graph=grasp_at_point)
    if output_file is None or output_file == "":
        name = os.path.splitext(os.path.basename(image_file))[0]
        output_file = os.path.join(os.getcwd(), name + "_grasp.png")
    output_file = os.fspath(output_file)
    if select_truss:
        return _select_interactively(scene, options, output_file)
    if truss_point is None:
        truss_point = np.full(2, np.nan)
    elif np.any(truss_point < 0.5) or np.any(truss_point > np.array([scene["w"], scene["h"]]) + 0.5):
        raise ValueError("truss_point lies outside the image.")
    point, direction, info = _detect_grasp(scene, truss_point, options, output_file)
    _write_output(_annotate_grasps(scene["I"], point[None, :], [info]), output_file, fast_png)
    return point, direction, info


def _analyze_scene(image_file, max_dimension, legacy=False, partition=False, point_graph=False):
    """Read the image and compute the masks, distance maps and skeleton graph once.

    partition computes the truss partition of the stem components, needed for
    a truss selection; point_graph computes the graph of the unpruned skeleton,
    needed for grasp_at_point.
    """
    sc = {}
    I = _read_rgb(image_file)
    h, w = I.shape[:2]
    scale = min(1.0, max_dimension / max(h, w))
    Ju = I if scale == 1 else _imresize(I, scale)
    hj, wj = Ju.shape[:2]
    J = Ju.astype(np.float64) / 255.0
    R, G, B = J[:, :, 0], J[:, :, 1], J[:, :, 2]
    V = J.max(axis=2)
    S = np.divide(V - J.min(axis=2), V, out=np.zeros_like(V), where=V != 0)
    # Yellow-green stems may have R slightly above G. Requiring a green-blue
    # contrast rejects the beige table, while saturation rejects neutral metal.
    color = (G > 1.12 * B) & (G - B > 0.035) & (S > 0.16) & (V > 0.12)
    strict = color & (G > 0.95 * R)
    stem = _clean_stem(strict)
    # Olive stems under warm light can fail G > 0.95 R along much of their
    # length, which breaks the peduncle and removes its free end. Pixels with
    # G > 0.8 R are then added if they are saturated (S > 0.3, which rejects
    # gray metal and paper) and connected to strict stem pixels. This is done
    # only if the strict mask covers less than 40% of the resulting stem area:
    # on the supplied images, it covered 31% on the one photo whose stems
    # fail, and at least 49% on all others.
    loose_stem = False
    if not legacy:
        loose = color & (G > 0.8 * R) & ((G > 0.95 * R) | (S > 0.3))
        grown = _clean_stem(_reconstruct(strict & loose, loose))
        if np.count_nonzero(stem) < 0.4 * np.count_nonzero(grown):
            stem = grown
            loose_stem = True
    holes = _fill_holes(stem) & ~stem
    stem = stem | (holes & ~_bwareaopen(holes, 40))
    # The truss selection keeps the fruit mask of the initial version, with all
    # holes filled: its path through touching tomatoes should not break at the
    # shadowed gaps between them. Only the selection and legacy use it.
    selection_fruit = None
    if legacy or partition:
        fruit = (R > 1.3 * G) & (R > 1.08 * B) & (R - G > 0.09)
        selection_fruit = _fill_holes(_imclose(_bwareaopen(fruit, 80), 3))
    if legacy:
        fruit = selection_fruit
    else:
        # Violet shadows between tomatoes, from warm or colored light, can pass
        # R > 1.08 B; ripe tomatoes have R several times B, so R > 1.5 B is required.
        fruit = (R > 1.3 * G) & (R > 1.5 * B) & (R - G > 0.09)
        fruit = _imclose(_bwareaopen(fruit, 80), 3)
        # Holes are calyxes and highlights on a fruit. Touching tomatoes also
        # enclose gaps that contain shadow and the peduncle; filling them would
        # put the peduncle inside the fruit mask. A hole is therefore filled only
        # if its part outside the stem mask is at most 0.25% of the image area.
        # On the supplied images, calyx and highlight holes stayed below 0.2%.
        holes = _fill_holes(fruit) & ~fruit
        n, hole_labels = cv2.connectedComponents(holes.view(np.uint8), connectivity=4)
        open_area = np.bincount(hole_labels[~stem], minlength=n)
        fill = open_area <= 0.0025 * fruit.size
        fill[0] = False
        fruit = fruit | fill[hole_labels]
    # A more inclusive mask recovers pale stem margins omitted by the strict
    # detection mask. Only the contiguous cross-section around this stem is
    # used, with a width limit to avoid absorbing neighboring objects.
    edge_mask = _imclose(color & (G > 0.92 * R), 1)
    # With the loose stem mask, the edge mask must contain the added stem pixels.
    if loose_stem:
        edge_mask = edge_mask | stem
    sc["edge_mask"] = edge_mask
    radius = _bwdist(~stem)
    fruit_distance = _bwdist(fruit)
    # Spurs up to 20 px are pruned at the default working size of 1600 px. In
    # smaller working images, the free peduncle end beyond the last junction can
    # be shorter than that, so the pruning length is scaled with the image size.
    prune_length = max(5, min(20, int(np.floor(20 * max(hj, wj) / 1600 + 0.5))))
    skeleton = _bwskel(stem, prune_length)
    # Column-major (MATLAB) linear indices keep candidate order, and therefore
    # tie-breaking in the ranking, identical to the MATLAB implementation.
    neighbors, degree, pixel_ids = _skeleton_graph(skeleton)
    end_nodes = np.flatnonzero(degree == 1)
    ends = pixel_ids[end_nodes]
    labels, n_labels = _label_column_major(stem)
    areas = np.bincount(labels.ravel(), minlength=n_labels + 1)
    # Olive or brownish stems under warm light can fail the green-over-red test
    # of the stem mask, so that one truss falls apart into many stem components.
    # A relaxed color test groups these components into trusses; the candidates
    # are still ranked on the strict stem mask. Only the selection uses it.
    relaxed_labels = None
    if partition:
        relaxed = (G > 0.75 * R) & (G > 1.2 * B) & (G - B > 0.05) & (S > 0.2) & (V > 0.12)
        relaxed_labels, _ = _label_column_major(_bwareaopen(_imclose(relaxed | stem, 1), 100))
    sc.update(I=I, h=h, w=w, hj=hj, wj=wj, xy_scale=np.array([w / wj, h / hj]),
              stem=stem, fruit=fruit, selection_fruit=selection_fruit, loose_stem=loose_stem,
              radius=radius, fruit_distance=fruit_distance,
              skeleton=skeleton, neighbors=neighbors, degree=degree, pixel_ids=pixel_ids,
              end_nodes=end_nodes, ends=ends, labels=labels, areas=areas,
              relaxed_labels=relaxed_labels,
              labels_f=labels.ravel(order="F"), radius_f=radius.ravel(order="F"),
              fruit_distance_f=fruit_distance.ravel(order="F"),
              end_xy=np.column_stack((ends // hj + 1.0, ends % hj + 1.0)))
    sc["end_radius"] = sc["radius_f"][ends]
    # With a truss selection, the stem components are partitioned into trusses
    # once per image, so that every click on the same truss selects the same set.
    sc["truss"] = _truss_partition(sc) if partition else None
    # grasp_at_point follows the stem on the unpruned skeleton: the free
    # peduncle end beyond the last junction can be shorter than the 20 px
    # pruning length in small images, and it is the part that a user points at.
    sc["point_graph"] = None
    if point_graph:
        sc["point_graph"] = _skeleton_graph(_bwskel(stem, 0))
    return sc


def _clean_stem(stem):
    """Close small gaps, remove thin protrusions and components below 100 px."""
    return _bwareaopen(_imopen(_imclose(stem, 1), 2), 100)


def _reconstruct(marker, mask):
    """Morphological reconstruction by dilation, 8-connected (MATLAB imreconstruct):
    the 8-connected components of mask that contain a marker pixel."""
    n, lab = cv2.connectedComponents(mask.view(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[lab[marker & mask]] = True
    keep[0] = False
    return keep[lab]


def _detect_grasp(sc, truss_point, options, output_file):
    """Rank the terminal branches, of the truss at truss_point if it is not NaN,
    and place the grasp on the best one. truss_point is in original pixels."""
    hj, wj, xy_scale = sc["hj"], sc["wj"], sc["xy_scale"]
    radius_f, fruit_distance = sc["radius_f"], sc["fruit_distance"]
    labels_f, areas, ends = sc["labels_f"], sc["areas"], sc["ends"]
    legacy = options["legacy"]
    selected = bool(np.all(np.isfinite(truss_point)))
    if selected and options["grasp_at_point"]:
        return _grasp_at_point(sc, truss_point, options, output_file)
    # With a selected truss, only the branches of its stem components compete.
    selected_label = np.nan
    members = None
    if selected:
        truss_work = (truss_point - 0.5) / xy_scale + 0.5
        selected_label = _select_truss_component(sc, truss_work)
        members = np.array([selected_label])
        if selected_label > 0 and sc["truss"] is not None:
            members = np.flatnonzero(sc["truss"] == sc["truss"][selected_label])
        # A truss of several components is identified by its smallest index, so
        # that clicks on different pieces of it select the same truss.
        selected_label = int(members.min())
    in_truss = np.zeros(sc["areas"].size, bool)
    if selected:
        in_truss[members] = True
    candidates, paths = [], []
    for k, e in enumerate(ends):
        if selected and not in_truss[labels_f[e]]:
            continue
        path = sc["pixel_ids"][_trace_terminal(sc["neighbors"], sc["degree"], sc["end_nodes"][k])]
        if path.size < 6:
            continue
        px = path // hj + 1.0
        py = path % hj + 1.0
        xy = np.column_stack((px, py))
        arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))))
        L = arc[-1]
        # Skeleton terminals of a blunt cylinder have a finite initial radius;
        # leaf/sepal tips start narrow and widen strongly towards their base.
        r_path = radius_f[path]
        r_body = np.median(r_path[(arc >= 0.25 * L) & (arc <= 0.75 * L)])
        diameter = 2 * r_body
        # A pedicel whose attachment is missing from the stem mask ends close to
        # the peduncle instead of joining it. The end of any other branch whose
        # gap to the stem surface is at most one diameter is treated as a junction.
        cut = None if legacy else _nearest_branch_end(
            xy, r_path, sc["end_xy"], sc["end_radius"], path[0], path[-1], ends, diameter)
        adjacent_end = np.full(2, np.nan)
        # The peduncle continues past a missed pedicel; the full branch is kept
        # for the direction fit, and only the free segment is truncated.
        full_xy, stop = xy, None
        if cut is not None:
            i_cut, adjacent_end = cut
            stop = i_cut
            path, xy, arc, r_path = path[:i_cut + 1], xy[:i_cut + 1], arc[:i_cut + 1], r_path[:i_cut + 1]
            L = arc[-1]
            if path.size < 6:
                continue
        r0 = r_path[arc <= np.float32(min(L / 3, diameter))].max()
        if L < 6 or diameter < 2 or areas[labels_f[e]] < 100:
            continue
        bluntness = min(1.0, r0 / max(r_body, 1.0))
        # Keep candidate selection independent of the requested grasp distance.
        inset = min(1.5 * diameter, 0.55 * L)
        p = _interp1(arc, xy, inset)
        # Fit the local stem tangent; orient it towards the terminal endpoint.
        fit_xy = xy[np.abs(arc - inset) <= max(4.0, diameter)]
        if fit_xy.shape[0] < 3:
            fit_xy = xy
        v = np.linalg.svd(fit_xy - fit_xy.mean(axis=0), full_matrices=False)[2][0]
        if np.dot(v, xy[0] - p) < 0:
            v = -v
        straightness = np.linalg.norm(xy[-1] - xy[0]) / max(L, 1.0)
        clearance = _interp_grid(fruit_distance, p[None, :], 0.0)[0]
        if not np.isfinite(clearance):
            clearance = 0.0
        # Emphasize thick, blunt ends; then prefer long, unobstructed branches.
        # No image-y term: "top" means exposed, not the upper image border.
        score = (diameter * bluntness ** 2 * straightness ** 2
                 * min(2.0, L / max(3 * diameter, 1.0))
                 * (0.5 + min(4.0, clearance / max(diameter, 1.0)) ** 2))
        # Pedicels attached to fruit can also look blunt, but their endpoint
        # touches the red fruit. A detached peduncle should have a free end.
        tip_clearance = sc["fruit_distance_f"][e]
        score *= min(1.0, max(0.05, tip_clearance / max(diameter, 1.0)))
        border = min(xy[0, 0] - 1, wj - xy[0, 0], xy[0, 1] - 1, hj - xy[0, 1])
        if border < diameter:
            score *= 0.15
        candidates.append(dict(point=p, direction=v, endpoint=xy[0].copy(),
                               score=float(score), diameter=float(diameter),
                               freeLength=float(L), bluntness=float(bluntness),
                               clearance=float(clearance), adjacentEnd=adjacent_end))
        paths.append((full_xy, stop))

    point = np.full(2, np.nan)
    direction = np.full(2, np.nan)
    info = _new_info(output_file, truss_point, selected_label)
    if candidates:
        # Python's sort is stable, as MATLAB's sort(...,'descend') is.
        order = sorted(range(len(candidates)), key=lambda i: -candidates[i]["score"])
        candidates = [candidates[i] for i in order]
        # Score per stem diameter: the product of the dimensionless factors of
        # the score, at most 33. Computed before the diameters are rescaled.
        info["relativeScore"] = float(candidates[0]["score"] / candidates[0]["diameter"])
        point, direction, placement = _place_on_stem(
            *paths[order[0]], candidates[0]["diameter"], sc["edge_mask"], sc["stem"],
            xy_scale, options["grasp_distance"], options["grasp_inset"], legacy)
        mean_scale = xy_scale.mean()
        for c in candidates:
            c["point"] = (c["point"] - 0.5) * xy_scale + 0.5
            c["endpoint"] = (c["endpoint"] - 0.5) * xy_scale + 0.5
            c["adjacentEnd"] = (c["adjacentEnd"] - 0.5) * xy_scale + 0.5
            v = c["direction"] * xy_scale
            c["direction"] = v / np.linalg.norm(v)
            c["diameter"] *= mean_scale
            c["freeLength"] *= mean_scale
            c["clearance"] *= mean_scale
        candidates[0]["point"] = point
        candidates[0]["direction"] = direction
        candidates[0]["endpoint"] = placement["cutEnd"]
        candidates[0]["freeLength"] = placement["freeLength"]
        info.update(placement)
        info["status"] = "best_candidate"
        if selected and info["relativeScore"] < 4:
            info["status"] = "weak_candidate"
            warnings.warn("The selected truss has no clearly exposed peduncle end (score "
                          f"per diameter {info['relativeScore']:.1f}); the grasp point may "
                          "lie on a pedicel.", RuntimeWarning, stacklevel=3)
        info["arrowEnd"] = point + options["arrow_length"] * direction
        info["candidates"] = candidates
    elif selected:
        warnings.warn("No stem terminal detected on the selected truss. Returning NaNs.",
                      RuntimeWarning, stacklevel=3)
    else:
        warnings.warn("No stem terminal detected. Returning NaNs and saving "
                      "the unmarked image.", RuntimeWarning, stacklevel=3)
    if options["save_diagnostics"]:
        _add_diagnostics(info, sc, members)
    return point, direction, info


def _new_info(output_file, truss_point, truss_component):
    """Result dict without a grasp; all modes return these keys."""
    return dict(status="no_candidate", outputFile=output_file,
                coordinateFrame="EXIF-oriented image; x right, y down",
                arrowEnd=np.full(2, np.nan), candidates=[],
                cutEnd=np.full(2, np.nan), firstJunction=np.full(2, np.nan),
                freeLength=np.nan, graspDistance=np.nan, distanceClamped=False,
                crossSectionEdges=np.full((2, 2), np.nan), centerline=np.zeros((0, 2)),
                trussPoint=np.asarray(truss_point, dtype=np.float64).copy(),
                trussComponent=truss_component, relativeScore=np.nan,
                accessibility="RGB visibility heuristic; physical height is unknown")


def _add_diagnostics(info, sc, members=None):
    """Store the working masks, and the mask of the selected truss's stem
    components (members; None without a selection)."""
    info["stemMask"] = sc["stem"]
    info["fruitMask"] = sc["fruit"]
    info["skeleton"] = sc["skeleton"]
    if members is not None:
        info["trussMask"] = np.isin(sc["labels"], members) & (info["trussComponent"] > 0)


def _annotate_grasps(I, points, infos):
    """Return a copy of I with each grasp and each truss-selection point drawn."""
    I = I.copy()
    for p, info in zip(points, infos):
        if info["status"] in ("best_candidate", "weak_candidate", "at_point"):
            weak = info["status"] == "weak_candidate"
            _draw_grasp(I, p, info["arrowEnd"], WEAK_COLOR if weak else ARROW_COLOR,
                        WEAK_COLOR if weak else POINT_RING_COLOR)
        if np.all(np.isfinite(info["trussPoint"])):
            _draw_selection(I, info["trussPoint"])
    return I


def _write_output(I, output_file, fast_png):
    extension = os.path.splitext(output_file)[1].lower()
    if fast_png and extension == ".png":
        write_tomato_png(I, output_file)
    elif not cv2.imwrite(output_file, np.ascontiguousarray(I[:, :, ::-1])):
        raise OSError(f"Could not write {output_file}.")


def _select_interactively(sc, options, output_file):
    """Let the user click trusses in a figure; show and save the result after each.

    The figure shows the original image. Clicks are converted to 1-based
    original-image pixels, the convention of all returned coordinates.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    height = 9.0
    fig = plt.figure(_FIGURE_NAME, figsize=(height * sc["w"] / sc["h"] + 0.8, height))
    fig.clf()
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
    handle = ax.imshow(sc["I"], interpolation="antialiased")
    ax.set_axis_off()
    file_name = os.path.basename(output_file)
    multiple = options["multiple_trusses"]
    ax.set_title("Click anywhere on a truss to be grasped; press Enter to finish" if multiple
                 else "Click anywhere on the truss to be grasped (Enter cancels)")
    fig.canvas.draw_idle()
    points, directions, infos = [], [], []
    while True:
        click = _wait_for_click(fig)
        if click is None:
            break
        if np.any(click < 0.5) or np.any(click > np.array([sc["w"], sc["h"]]) + 0.5):
            continue  # Outside the image.
        p, v, info = _detect_grasp(sc, click, options, output_file)
        if multiple:
            # A click on an already labelled truss replaces its label.
            keep = [i for i, old in enumerate(infos)
                    if old["trussComponent"] != info["trussComponent"]]
            points = [points[i] for i in keep]
            directions = [directions[i] for i in keep]
            infos = [infos[i] for i in keep]
            points.append(p)
            directions.append(v)
            infos.append(info)
        else:
            points, directions, infos = [p], [v], [info]
        annotated = _annotate_grasps(sc["I"], points, infos)
        _write_output(annotated, output_file, options["fast_png"])
        if not plt.fignum_exists(fig.number):
            break
        handle.set_data(annotated)
        status = info["status"].replace("_", " ")
        if multiple:
            ax.set_title(f"{len(infos)} truss(es) labelled, last: {status}. "
                         "Click another truss, or press Enter to finish.")
        else:
            ax.set_title(f"Result: {status}. Saved to {file_name}")
        fig.canvas.draw_idle()
        if not multiple:
            break
    if not infos:
        raise RuntimeError("No truss was selected.")
    if multiple and plt.fignum_exists(fig.number):
        ax.set_title(f"{len(infos)} truss(es) labelled. Saved to {file_name}")
        fig.canvas.draw_idle()
    if matplotlib.get_backend().lower() != "agg":
        plt.pause(0.001)  # Render the final state; no-op without a window.
    if multiple:
        return np.array(points), np.array(directions), infos
    return points[0], directions[0], infos[0]


def _wait_for_click(fig):
    """One left click in fig as 1-based [x, y] pixels, or None for Enter or close."""
    import matplotlib.pyplot as plt
    if not plt.fignum_exists(fig.number):
        return None
    try:
        clicks = fig.ginput(1, timeout=0)
    except Exception:  # The figure was closed while waiting.
        return None
    if not clicks:
        return None
    # imshow places pixel centers at 0-based integer coordinates.
    return np.array(clicks[0], dtype=np.float64) + 1.0


def _read_rgb(image_file):
    """Return the image as uint8 RGB (h, w, 3), EXIF-oriented for JPEG/TIFF."""
    with Image.open(image_file) as im:
        # JPEG/TIFF viewers apply EXIF orientation; PNG output has no such tag.
        # Normalize pixels BEFORE detection, so every point and vector is
        # expressed directly in the displayed coordinate frame.
        if im.format in ("JPEG", "MPO", "TIFF"):
            im = ImageOps.exif_transpose(im)
        if im.mode == "P":
            im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        if im.mode in ("RGBA", "RGBX", "RGBa"):
            im = im.convert("RGBA")
            return np.ascontiguousarray(np.asarray(im)[:, :, :3])
        if im.mode == "RGB":
            # A writable copy, since the annotation is drawn in place.
            return np.array(im)
        bands = im.getbands()
    if len(bands) >= 3:
        # 16-bit or other multi-band formats: let OpenCV decode them.
        data = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        if data is not None and data.ndim == 3 and data.shape[2] >= 3:
            data = data[:, :, 2::-1]
            if data.dtype == np.uint16:
                data = np.round(data / 257.0).astype(np.uint8)
            if data.dtype == np.uint8:
                return np.ascontiguousarray(data)
    raise ValueError("Input must be an RGB color image.")


def _imresize(I, scale):
    """Reproduce MATLAB imresize(I, scale) for uint8 RGB: antialiased bicubic."""
    h, w = I.shape[:2]
    # MATLAB resizes dimension 1 (rows) first when both scales are equal, and
    # rounds to uint8 after each pass.
    wr, ir = _contributions(h, int(np.ceil(scale * h)), scale)
    wc, ic = _contributions(w, int(np.ceil(scale * w)), scale)
    rows = _resize_rows(I.reshape(h, -1), wr, ir).reshape(-1, w, 3)
    return _resize_cols(rows, wc, ic)


def _contributions(n_in, n_out, scale, kernel_width=4.0):
    """Return the (n_out, P) weights and 0-based input indices of MATLAB imresize."""
    if scale < 1:
        kernel = lambda x: scale * _cubic(scale * x)
        kernel_width = kernel_width / scale
    else:
        kernel = _cubic
    x = np.arange(1, n_out + 1, dtype=np.float64)[:, None]
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(np.ceil(kernel_width)) + 2
    indices = left + np.arange(P)
    weights = kernel(u - indices)
    weights = weights / weights.sum(axis=1, keepdims=True)
    aux = np.concatenate((np.arange(n_in), np.arange(n_in - 1, -1, -1)))
    indices = aux[np.mod(indices - 1, aux.size).astype(np.int64)]
    return np.ascontiguousarray(weights), np.ascontiguousarray(indices)


@numba.njit(parallel=True, cache=True)
def _resize_rows(A, weights, indices):
    n_out, P = weights.shape
    m = A.shape[1]
    out = np.empty((n_out, m), np.uint8)
    for i in numba.prange(n_out):
        acc = np.zeros(m)
        for k in range(P):
            wk = weights[i, k]
            row = A[indices[i, k]]
            for j in range(m):
                acc[j] += wk * row[j]
        for j in range(m):
            out[i, j] = min(255.0, max(0.0, np.floor(acc[j] + 0.5)))
    return out


@numba.njit(parallel=True, cache=True)
def _resize_cols(A, weights, indices):
    h = A.shape[0]
    n_out, P = weights.shape
    out = np.empty((h, n_out, 3), np.uint8)
    for r in numba.prange(h):
        for j in range(n_out):
            for c in range(3):
                acc = 0.0
                for k in range(P):
                    acc += weights[j, k] * A[r, indices[j, k], c]
                out[r, j, c] = min(255.0, max(0.0, np.floor(acc + 0.5)))
    return out


def _cubic(x):
    ax = np.abs(x)
    ax2 = ax ** 2
    ax3 = ax ** 3
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1)
            + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((1 < ax) & (ax <= 2)))


def _imclose(mask, r):
    # MATLAB imclose zero-pads by ceil(size(se)/2) first, so pixels outside
    # the image count as background during the erosion.
    p = r + 1
    padded = cv2.copyMakeBorder(mask.view(np.uint8), p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, _DISK[r])
    return np.ascontiguousarray(closed[p:-p, p:-p]).view(bool)


def _imopen(mask, r):
    # MATLAB imopen does not pad: outside pixels are foreground for the
    # erosion and background for the dilation, as in OpenCV's defaults.
    return cv2.morphologyEx(mask.view(np.uint8), cv2.MORPH_OPEN, _DISK[r]).view(bool)


def _bwareaopen(mask, p):
    """Remove 8-connected components with fewer than p pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= p
    keep[0] = False
    return keep[labels]


def _fill_holes(mask):
    """Fill holes, i.e. 4-connected background regions not touching the border."""
    n, labels = cv2.connectedComponents((~mask).view(np.uint8), connectivity=4)
    border = np.zeros(n, bool)
    for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        border[edge] = True
    border[0] = True
    return mask | ~border[labels]


def _bwdist(mask):
    """Return the Euclidean distance of every pixel to the nearest True pixel."""
    # Single precision, as MATLAB bwdist returns; the scores inherit it.
    # OpenCV's precise mode is an exact Euclidean transform; it gave values
    # identical to scipy.ndimage.distance_transform_edt on all test masks.
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform((~mask).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _bwskel(mask, min_branch_length):
    """Reproduce MATLAB bwskel(mask,'MinBranchLength',n) for 2-D images."""
    img = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    img[1:-1, 1:-1] = mask
    skel = _lee_thin(img, _THIN_DIRECTIONS, _SIMPLE_LUT, _CANDIDATE_LUT)[1:-1, 1:-1].view(bool)
    if min_branch_length > 0:
        skel = _prune_edges(skel, min_branch_length)
    return skel


def _thinning_luts():
    """Return 256-entry lookup tables over the 8-neighborhood code of a pixel.

    MATLAB bwskel applies the 3-D thinning of Lee, Kashyap and Chu (1994) to
    the image as a single padded slice. For such planar sets, a (26,6) simple
    point is a pixel whose 8-neighbors form one 8-connected component, and
    Euler invariance reduces to the 2-D (8,4) Euler number, evaluated with
    Gray's bit-quad formula on the four 2x2 quads that contain the pixel.
    A deletion candidate must also not be an endpoint (exactly one neighbor).
    """
    simple = np.zeros(256, np.bool_)
    candidate = np.zeros(256, np.bool_)
    positions = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]

    def quad_euler(q):
        s = q.sum()
        if s == 1:
            return 1
        if s == 3:
            return -1
        if s == 2 and q[0, 0] == q[1, 1]:
            return -2
        return 0

    for code in range(256):
        g = np.zeros((3, 3), np.int64)
        for k, (r, c) in enumerate(positions):
            g[r, c] = code >> k & 1
        points = [p for p in positions if g[p]]
        seen, components = set(), 0
        for p in points:
            if p in seen:
                continue
            components += 1
            stack = [p]
            seen.add(p)
            while stack:
                a = stack.pop()
                for q in points:
                    if q not in seen and max(abs(q[0] - a[0]), abs(q[1] - a[1])) == 1:
                        seen.add(q)
                        stack.append(q)
        with_p, without_p = g.copy(), g.copy()
        with_p[1, 1] = 1
        change = sum(quad_euler(with_p[r:r + 2, c:c + 2]) - quad_euler(without_p[r:r + 2, c:c + 2])
                     for r in (0, 1) for c in (0, 1))
        simple[code] = components == 1
        candidate[code] = len(points) != 1 and change == 0 and components == 1
    return simple, candidate


_SIMPLE_LUT, _CANDIDATE_LUT = _thinning_luts()
# Border directions (row, column offset of the neighbor that must be background)
# in the order and orientation of MATLAB's bwskel. (0, 0) denotes the two
# out-of-plane directions, for which every pixel of a single slice is a border point.
_THIN_DIRECTIONS = np.array([(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), (0, 0)], np.int64)


@numba.njit(cache=True)
def _neighbor_code(img, r, c):
    # Bit order follows the row-major neighbor positions of _thinning_luts.
    code = 0
    if img[r - 1, c - 1]:
        code |= 1
    if img[r - 1, c]:
        code |= 2
    if img[r - 1, c + 1]:
        code |= 4
    if img[r, c - 1]:
        code |= 8
    if img[r, c + 1]:
        code |= 16
    if img[r + 1, c - 1]:
        code |= 32
    if img[r + 1, c]:
        code |= 64
    if img[r + 1, c + 1]:
        code |= 128
    return code


@numba.njit(cache=True)
def _lee_thin(img, directions, simple, candidate):
    """Thin a zero-padded uint8 image in place, as MATLAB bwskel does."""
    H, W = img.shape
    rows = np.empty(H * W, np.int64)
    cols = np.empty(H * W, np.int64)
    unchanged = 0
    while unchanged < 6:
        unchanged = 0
        for d in range(6):
            dr, dc = directions[d, 0], directions[d, 1]
            n = 0
            # Candidates are collected in parallel, in column-major order...
            for c in range(1, W - 1):
                for r in range(1, H - 1):
                    if img[r, c] == 0:
                        continue
                    if (dr != 0 or dc != 0) and img[r + dr, c + dc] != 0:
                        continue
                    if candidate[_neighbor_code(img, r, c)]:
                        rows[n] = r
                        cols[n] = c
                        n += 1
            # ...then deleted sequentially, re-checking that each is still simple.
            no_change = True
            for i in range(n):
                r, c = rows[i], cols[i]
                img[r, c] = 0
                if simple[_neighbor_code(img, r, c)]:
                    no_change = False
                else:
                    img[r, c] = 1
            if no_change:
                unchanged += 1
    return img


def _prune_edges(skel, thresh):
    """Port of MATLAB images.internal.pruneEdges3 for a 2-D skeleton.

    Links that end in an endpoint and contain no more than thresh pixels
    (both terminal pixels included) are removed. The image is traversed in
    column-major order, as in MATLAB, so the same links are kept.
    """
    if not skel.any():
        return skel
    h, w = skel.shape
    ph = h + 2
    padded = np.zeros((ph, w + 2), bool)
    padded[1:-1, 1:-1] = skel
    flat = padded.ravel(order="F")
    pts = np.flatnonzero(flat)
    # 3x3 neighborhood offsets in MATLAB order (row index fastest).
    offs = np.array([dx * ph + dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    nh_idx = pts[:, None] + offs[None, :]
    nh = flat[nh_idx]
    sum_nh = nh.sum(axis=1)
    nodes = pts[sum_nh > 3]
    if nodes.size == 0:
        return skel
    ep = pts[sum_nh == 2]
    can_mask = sum_nh == 3
    cans = pts[can_mask]
    can_nb = np.where(nh[can_mask], nh_idx[can_mask], 0)
    can_nb = np.delete(can_nb, 4, axis=1)
    can_nb = np.sort(can_nb, axis=1)[:, -2:]

    label = flat.astype(np.int64)
    node_pixels, node_is_ep = [], []
    for group, is_ep in ((nodes, False), (ep, True)):
        g = np.zeros(padded.size, bool)
        g[group] = True
        g = g.reshape(padded.shape, order="F")
        n, lab = cv2.connectedComponents(g.view(np.uint8), connectivity=8)
        lab_f = lab.ravel(order="F")
        members = np.flatnonzero(lab_f)
        comp = lab_f[members]
        # bwconncomp numbers components by their first pixel in column-major
        # order, whereas OpenCV scans row-major; renumber accordingly.
        first = np.full(n, np.iinfo(np.int64).max)
        np.minimum.at(first, comp, members)
        rank = np.empty(n, np.int64)
        rank[np.argsort(first[1:]) + 1] = np.arange(n - 1)
        comp = rank[comp]
        sort_idx = np.lexsort((members, comp))
        members, comp = members[sort_idx], comp[sort_idx]
        splits = np.flatnonzero(np.diff(comp)) + 1
        for part in np.split(members, splits):
            node_pixels.append(part)
            node_is_ep.append(is_ep)
    for i, part in enumerate(node_pixels):
        label[part] = i + 2

    c2n = {int(c): j for j, c in enumerate(cans)}
    s2n = {int(p): j for j, p in enumerate(pts)}
    nh_list = nh.tolist()
    nh_idx_list = nh_idx.tolist()
    can_nb_list = can_nb.tolist()
    node_links = [0] * len(node_pixels)
    kept = []
    for i, part in enumerate(node_pixels):
        for start in part.tolist():
            row = s2n[start]
            for c, present in zip(nh_idx_list[row], nh_list[row]):
                if not present or label[c] != 1:
                    continue
                vox = [start]
                idx = c
                previous = start
                while True:
                    a, b = can_nb_list[c2n[idx]]
                    nxt = b if a == previous else a
                    vox.append(idx)
                    if label[nxt] > 1:
                        vox.append(nxt)
                        n_idx = label[nxt] - 2
                        break
                    previous, idx = idx, nxt
                label[vox[1:-1]] = 0
                is_ep = node_is_ep[n_idx]
                if (is_ep and len(vox) > thresh) or not is_ep:
                    node_links[i] += 1
                    node_links[n_idx] += 1
                    kept.append(vox)
    out = np.zeros(padded.size, bool)
    for i, part in enumerate(node_pixels):
        if node_links[i]:
            out[part] = True
    for vox in kept:
        out[vox] = True
    return out.reshape(padded.shape, order="F")[1:-1, 1:-1]


def _nearest_branch_end(xy, radius, end_xy, end_radius, own_start, own_stop, end_ids, diameter):
    """Find the first branch pixel that has the blunt end of another branch beside it.

    An endpoint counts if the gap between the two stem surfaces, i.e., its
    distance to a branch pixel minus both radii, is at most one diameter, and if
    its radius is at least 0.2 diameters. The second condition keeps a broken-off
    pedicel, which ends bluntly, and rejects sepal tips, which taper to a point.
    Returns (index of the nearest branch pixel, endpoint [x, y]) for the endpoint
    nearest to the start of the branch, or None.

    Args:
        xy: (n, 2) 1-based [x, y] of the branch pixels, starting at its endpoint.
        radius: (n,) distance-transform radius at the branch pixels, pixels.
        end_xy: (m, 2) 1-based [x, y] of all skeleton endpoints.
        end_radius: (m,) distance-transform radius at the endpoints, pixels.
        own_start, own_stop: pixel ids of the first and last branch pixels.
        end_ids: (m,) pixel ids of the endpoints, to exclude the branch's own ends.
        diameter: stem diameter of the branch, pixels.
    """
    reach = radius.max() + diameter
    lo, hi = xy.min(axis=0) - reach, xy.max(axis=0) + reach
    near = (np.all(end_xy >= lo, axis=1) & np.all(end_xy <= hi, axis=1)
            & (end_ids != own_start) & (end_ids != own_stop)
            & (end_radius >= 0.2 * diameter))
    if not near.any():
        return None
    points = end_xy[near]
    gaps = (np.linalg.norm(points[:, None, :] - xy[None, :, :], axis=2)
            - radius[None, :] - end_radius[near][:, None])
    nearest = gaps.argmin(axis=1)
    hits = gaps[np.arange(points.shape[0]), nearest] <= diameter
    if not hits.any():
        return None
    k = np.flatnonzero(hits)[np.argmin(nearest[hits])]
    return int(nearest[k]), points[k].copy()


def _skeleton_graph(skeleton):
    """Return neighbor lists, degrees and column-major ids of skeleton pixels.

    Diagonal shortcuts are suppressed when an orthogonal connecting pixel
    exists. Counting all eight neighbors makes staircase corners into false
    junctions and can turn a blunt end into a three-pixel cycle with no endpoint.
    """
    h, w = skeleton.shape
    sk_t = skeleton.T
    pixel_ids = np.flatnonzero(sk_t)
    n = pixel_ids.size
    ids = np.zeros((h, w), np.int64)
    x = pixel_ids // h
    y = pixel_ids % h
    ids[y, x] = np.arange(1, n + 1)
    src, dst = [], []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
        valid = (x + dx >= 0) & (x + dx < w) & (y + dy < h)
        u = np.flatnonzero(valid)
        xx, yy = x[valid], y[valid]
        v = ids[yy + dy, xx + dx]
        keep = v > 0
        if dx != 0 and dy != 0:
            keep &= ~skeleton[yy, xx + dx] & ~skeleton[yy + dy, xx]
        src.append(u[keep])
        dst.append(v[keep] - 1)
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    A = sparse.csc_matrix((np.ones(2 * src.size), (np.concatenate((src, dst)),
                                                 np.concatenate((dst, src)))),
                          shape=(n, n))
    A.sort_indices()
    degree = np.diff(A.indptr)
    neighbors = np.split(A.indices, A.indptr[1:-1])
    return neighbors, degree, pixel_ids


def _trace_terminal(neighbors, degree, start):
    """Follow one skeleton edge until a junction. Do not jump across occlusions."""
    path = [start]
    previous, current = -1, start
    n = len(neighbors)
    while len(path) < n:
        nb = [j for j in neighbors[current] if j != previous]
        if len(nb) != 1:
            break
        previous, current = current, nb[0]
        path.append(current)
        if degree[current] != 2:
            break
    return np.array(path)


def _place_on_stem(xy, stop, diameter, mask, strict_mask, xy_scale, grasp_distance, grasp_inset,
                   legacy=False):
    """Build a smooth, width-centered curve, extending it to the visible cut face.

    xy is the full branch; stop is the index of the branch pixel at which the free
    segment ends (an adjacent branch end), or None when it ends at the junction.
    """
    mask = mask.astype(np.float64)
    strict_mask = strict_mask.astype(np.float64)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    start = min(diameter, 0.15 * arc[-1])
    offsets = _section_offsets(diameter)
    curve = _centered_curve(xy, start, diameter, offsets, mask, strict_mask, single=True)
    curve[-1] = xy[-1]  # Keep the detected side-stem junction anchored.
    # Recover the cut face, rather than measuring from an inset skeleton tip.
    t = curve[min(max(5, int(np.ceil(diameter / 2))), curve.shape[0]) - 1] - curve[0]
    t = t / np.linalg.norm(t)
    distances = _colon(0.0, 0.2, max(3 * diameter, 2 * start))
    ray = curve[0] - distances[:, None] * t
    values = _interp_grid(mask, ray, np.nan)
    values[np.isnan(values)] = 0
    outside = np.flatnonzero(values < 0.5)
    if outside.size == 0 or outside[0] == 0:
        cut = xy[0]
    else:
        o = outside[0]
        a = o - 1
        d = distances[a] + (0.5 - values[a]) / (values[o] - values[a]) * 0.2
        cut = curve[0] - d * t
    curve = np.vstack((cut, curve))
    curve = (curve - 0.5) * xy_scale + 0.5
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))))
    arc, unique_rows = np.unique(arc, return_index=True)
    curve = curve[unique_rows]
    full_arc, full_curve = arc, curve
    if stop is not None:
        # End the free segment at the curve point nearest to the truncation pixel.
        q = (xy[stop] - 0.5) * xy_scale + 0.5
        L_stop = full_arc[np.argmin(np.linalg.norm(full_curve - q, axis=1))]
        keep = arc < L_stop
        arc = np.concatenate((arc[keep], [L_stop]))
        curve = np.vstack((curve[keep], _interp1(full_arc, full_curve, L_stop)))
    L = arc[-1]
    mean_scale = xy_scale.mean()
    requested = grasp_distance
    if grasp_inset is not None:
        requested = grasp_inset * diameter * mean_scale
    if requested is None:
        requested = L / 2
    distance = min(requested, 0.9 * L)
    p = _interp1(arc, curve, distance)
    if legacy:
        span = max(1.0, 0.5 * diameter * mean_scale)
        t = (_interp1(arc, curve, min(L, distance + span))
             - _interp1(arc, curve, max(0.0, distance - span)))
        v = -t / np.linalg.norm(t)
    else:
        v = _grasp_direction(full_arc, full_curve, distance, diameter * mean_scale)
    # Final transverse correction uses the actual local tangent at the grasp.
    pw = (p - 0.5) / xy_scale + 0.5
    tw = -v / xy_scale
    tw = tw / np.linalg.norm(tw)
    pw, edges = _center_section(pw, tw, diameter, offsets, mask, strict_mask)
    p = (pw - 0.5) * xy_scale + 0.5
    details = dict(cutEnd=curve[0].copy(), firstJunction=curve[-1].copy(),
                   freeLength=float(L), graspDistance=float(distance),
                   distanceClamped=bool(requested > 0.9 * L),
                   crossSectionEdges=(edges - 0.5) * xy_scale + 0.5,
                   centerline=curve.copy())
    return p, v, details


def _centered_curve(xy, start, diameter, offsets, mask, strict_mask, single):
    """Resample the skeleton path xy from arc length start at about 2 px, smooth
    it, and center it twice between the stem edges.

    single=True reproduces MATLAB's single-precision sample positions when
    start derives from the single-precision diameter: when the last sample
    rounds above the double arc length, interp1 returns NaN and smoothdata
    omits that row.
    """
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    s = np.linspace(start, arc[-1], max(8, int(np.ceil((arc[-1] - start) / 2))))
    if single:
        s = s.astype(np.float32).astype(np.float64)
    curve = _movmean(_interp1(arc, xy, s), 9)
    for _ in range(2):
        span = int(max(3, np.ceil(diameter / 4)))
        rows = np.arange(curve.shape[0])
        m = curve.shape[0] - 1
        tangents = curve[np.minimum(m, rows + span)] - curve[np.maximum(0, rows - span)]
        for j in range(curve.shape[0]):
            t = tangents[j] / max(np.linalg.norm(tangents[j]), _EPS)
            curve[j] = _center_section(curve[j], t, diameter, offsets, mask, strict_mask)[0]
        curve = _movmean(curve, 9)
    return curve


def _grasp_direction(arc, curve, distance, d):
    """Return the unit stem direction at arc length distance, pointing to the cut face.

    A least-squares line is fitted to the centered curve over +-2d around the grasp
    point. A local difference over +-d/2 follows humps of the centerline caused by
    irregular mask edges. The first and last stem diameter of the free end are left
    out, because an oblique cut face and the flare of the junction bias the centering
    there; at least +-d/2 around the grasp point is kept for short free ends.

    Args:
        arc: (n,) arc length of the curve from the cut face, original pixels.
        curve: (n, 2) centered curve, original-image [x, y].
        distance: arc length of the grasp point, original pixels.
        d: stem diameter, original pixels.
    """
    L = arc[-1]
    lo = max(distance - 2 * d, min(d, distance - 0.5 * d), 0.0)
    hi = min(distance + 2 * d, max(L - d, distance + 0.5 * d), L)
    if hi - lo < 1.0:
        lo, hi = max(0.0, distance - 1.0), min(L, distance + 1.0)
    # np.interp clamps to the curve ends. MATLAB computes the limits in single
    # precision, and a limit that rounds just outside the curve is replaced by
    # the curve end there as well.
    query = np.linspace(lo, hi, max(3, int(np.ceil(hi - lo)) + 1))
    samples = np.column_stack([np.interp(query, arc, curve[:, c]) for c in range(2)])
    v = np.linalg.svd(samples - samples.mean(axis=0), full_matrices=False)[2][0]
    if np.dot(v, samples[0] - samples[-1]) < 0:
        v = -v
    return v / np.linalg.norm(v)


def _section_offsets(diameter):
    offsets = _colon(-2 * diameter, 0.2, 2 * diameter)
    return np.unique(np.concatenate((offsets, [0.0])))


def _center_section(p, t, diameter, offsets, mask, strict_mask):
    """Interpolate both edge crossings and take their midpoint across the width."""
    n = np.array([-t[1], t[0]])
    q = p + offsets[:, None] * n
    edges = np.vstack((p, p))
    middle = int(np.argmin(np.abs(offsets)))
    for m in (mask, strict_mask):
        values = _interp_grid(m, q, np.nan)
        values[np.isnan(values)] = 0
        if values[middle] < 0.5:
            continue
        below = values < 0.5
        left_hits = np.flatnonzero(below[:middle + 1])
        right_hits = np.flatnonzero(below[middle:])
        if left_hits.size == 0 or right_hits.size == 0:
            continue
        left = left_hits[-1]
        right = middle + right_hits[0]
        a = offsets[left] + (0.5 - values[left]) / (values[left + 1] - values[left]) * \
            (offsets[left + 1] - offsets[left])
        b = offsets[right - 1] + (0.5 - values[right - 1]) / (values[right] - values[right - 1]) * \
            (offsets[right] - offsets[right - 1])
        if b - a > 1.8 * diameter or abs((a + b) / 2) > 0.35 * diameter:
            continue
        edges = np.vstack((p + a * n, p + b * n))
        return edges.mean(axis=0), edges
    return p, edges


def _interp_grid(image, xy, fill):
    """Bilinear interpolation of image at 1-based [x, y] points (MATLAB interp2)."""
    h, w = image.shape
    x = xy[:, 0] - 1.0
    y = xy[:, 1] - 1.0
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    out = np.full(x.shape, fill, dtype=np.float64)
    if not inside.any():
        return out
    xi, yi = x[inside], y[inside]
    x0 = np.minimum(np.floor(xi).astype(np.int64), max(w - 2, 0))
    y0 = np.minimum(np.floor(yi).astype(np.int64), max(h - 2, 0))
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = xi - x0
    fy = yi - y0
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    out[inside] = top * (1 - fy) + bottom * fy
    return out


def _interp1(x, y, q):
    """Linear interpolation of the rows of y (n, 2) at q; NaN outside [x0, xn]."""
    q = np.asarray(q, dtype=np.float64)
    res = np.column_stack([np.interp(q.ravel(), x, y[:, c]) for c in range(y.shape[1])])
    res[(q.ravel() < x[0]) | (q.ravel() > x[-1])] = np.nan
    return res[0] if q.ndim == 0 else res


def _movmean(a, k):
    """Centered moving mean, shrinking end windows and omitting NaN (smoothdata)."""
    n = a.shape[0]
    half = (k - 1) // 2
    valid = ~np.isnan(a)
    zero = np.zeros((1, a.shape[1]))
    c = np.vstack((zero, np.cumsum(np.where(valid, a, 0.0), axis=0)))
    cn = np.vstack((zero, np.cumsum(valid, axis=0)))
    lo = np.maximum(0, np.arange(n) - half)
    hi = np.minimum(n, np.arange(n) + half + 1)
    count = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (c[hi] - c[lo]) / count, np.nan)


def _colon(a, d, b):
    """Return MATLAB's a:d:b, including its endpoint tolerance and symmetry."""
    if d == 0 or (a < b and d < 0) or (b < a and d > 0):
        return np.zeros(0)
    tol = 2.0 * _EPS * max(abs(a), abs(b))
    sig = np.sign(d)
    n = int(np.round((b - a) / d))
    if sig * (a + n * d - b) > tol:
        n -= 1
    c = a + n * d
    if sig * (c - b) > -tol:
        c = b
    y = np.zeros(n + 1)
    k = np.arange(n // 2 + 1)
    y[k] = a + k * d
    y[n - k] = c - k * d
    if n % 2 == 0:
        y[n // 2] = (a + c) / 2
    return y


def _draw_grasp(I, p, q, arrow_color=ARROW_COLOR, ring_color=POINT_RING_COLOR):
    """Draw the grasp marker and arrow in place (no GUI or CV toolbox needed)."""
    line_width = max(2, int(np.floor(max(I.shape[0], I.shape[1]) / 650 + 0.5)))
    v = (q - p) / np.linalg.norm(q - p)
    n = np.array([-v[1], v[0]])
    head = min(np.linalg.norm(q - p) * 0.3, 12 * line_width)
    for a, b in ((p, q), (q, q - head * v + 0.5 * head * n), (q, q - head * v - 0.5 * head * n)):
        _paint_line(I, a, b, line_width, arrow_color)
    _paint_disk(I, p, 4 * line_width, POINT_OUTLINE_COLOR)
    _paint_disk(I, p, 3 * line_width, ring_color)
    _paint_disk(I, p, line_width, POINT_CENTER_COLOR)


def _paint_line(I, a, b, r, color):
    h, w = I.shape[:2]
    x = np.arange(max(1, np.floor(min(a[0], b[0]) - r)), min(w, np.ceil(max(a[0], b[0]) + r)) + 1)
    y = np.arange(max(1, np.floor(min(a[1], b[1]) - r)), min(h, np.ceil(max(a[1], b[1]) + r)) + 1)
    if x.size == 0 or y.size == 0:
        return
    xx, yy = np.meshgrid(x, y)
    d = b - a
    t = np.clip(((xx - a[0]) * d[0] + (yy - a[1]) * d[1]) / np.sum(d ** 2), 0, 1)
    inside = (xx - a[0] - t * d[0]) ** 2 + (yy - a[1] - t * d[1]) ** 2 <= r ** 2
    I[yy[inside].astype(np.int64) - 1, xx[inside].astype(np.int64) - 1] = color


def _paint_disk(I, p, r, color):
    h, w = I.shape[:2]
    x = np.arange(max(1, np.floor(p[0] - r)), min(w, np.ceil(p[0] + r)) + 1)
    y = np.arange(max(1, np.floor(p[1] - r)), min(h, np.ceil(p[1] + r)) + 1)
    if x.size == 0 or y.size == 0:
        return
    xx, yy = np.meshgrid(x, y)
    inside = (xx - p[0]) ** 2 + (yy - p[1]) ** 2 <= r ** 2
    I[yy[inside].astype(np.int64) - 1, xx[inside].astype(np.int64) - 1] = color


def _label_column_major(mask):
    """Label 8-connected components, numbered as MATLAB's bwconncomp does.

    bwconncomp numbers components by their first pixel in column-major order,
    whereas OpenCV scans row-major. Returns (labels, number of components).
    """
    n, lab = cv2.connectedComponents(mask.view(np.uint8), connectivity=8)
    if n <= 1:
        return lab.astype(np.int64), 0
    flat = lab.ravel(order="F")
    first = np.full(n, np.iinfo(np.int64).max)
    np.minimum.at(first, flat, np.arange(flat.size))
    rank = np.zeros(n, np.int64)
    rank[np.argsort(first[1:]) + 1] = np.arange(1, n)
    return rank[lab], n - 1


def _select_truss_component(sc, point):
    """Select the stem component of the truss indicated by point ([x, y], 1-based
    working pixels).

    A click may fall on a fruit rather than on a stem. The calyxes lie on top
    of the fruit, so the convex hull of a truss's stem system covers most of its
    fruit. The component whose hull contains the point is chosen, the nearest
    one if several hulls contain it, and the nearest component otherwise.
    Returns 0 if there is no component large enough to hold a candidate. As in
    MATLAB's regionprops, the hull is taken over the midpoints of the pixel edges.

    A calyx or pedicel that is disconnected from its peduncle in the stem mask
    forms a fragment component, which lies on its fruit and therefore contains
    a click on that fruit. A fragment spans about one or two fruits, whereas the
    stem system of a truss spans several. A component is taken as a fragment if
    its maximum Feret diameter is below 0.2 times, and its area below 0.1 times,
    that of the largest component. If the selected component is a fragment, the
    selection is repeated among the other components; if none of their hulls
    contains the point, the one nearest along a path through the fruit and stem
    masks is chosen, since the fruits of a truss touch each other. If no other
    component is reachable, the fragment is kept. The thresholds were set on the
    supplied images, where fragments reached 0.17 and 0.065, and trusses at
    least 0.25 and 0.11.
    """
    labels, areas = sc["labels"], sc["areas"]
    valid = np.flatnonzero(areas >= 100)
    valid = valid[valid > 0]
    if valid.size == 0:
        return 0
    rows, cols = np.nonzero(labels)
    lab = labels[rows, cols]
    distance = np.full(valid.size, np.inf)
    inside = np.zeros(valid.size, bool)
    q = (float(point[0]), float(point[1]))
    coords = []
    for i, label in enumerate(valid):
        sel = lab == label
        x = cols[sel] + 1.0
        y = rows[sel] + 1.0
        coords.append((x, y))
        distance[i] = np.min(np.hypot(x - q[0], y - q[1]))
        edges = np.concatenate((np.column_stack((x - 0.5, y)), np.column_stack((x + 0.5, y)),
                                np.column_stack((x, y - 0.5)), np.column_stack((x, y + 0.5))))
        hull = cv2.convexHull(edges.astype(np.float32))
        inside[i] = cv2.pointPolygonTest(hull, q, False) >= 0
    label = _nearest_containing(valid, distance, inside)
    # With a truss partition, detached fragments already belong to their truss.
    if sc.get("truss") is not None:
        return label
    extent = np.array([_max_feret_diameter(x, y) for x, y in coords])
    long = (extent >= 0.2 * extent.max()) | (areas[valid] >= 0.1 * areas[valid].max())
    if long[valid == label][0] or not long.any():
        return label
    if inside[long].any():
        return _nearest_containing(valid[long], distance[long], inside[long])
    # Geodesic distance from the point through the fruit and stem masks.
    region = (sc["fruit"] if sc["selection_fruit"] is None else sc["selection_fruit"]) | sc["stem"]
    seed = np.floor(np.asarray(point, dtype=np.float64) + 0.5).astype(np.int64)
    seed = np.clip(seed, [1, 1], [sc["wj"], sc["hj"]])
    if not region[seed[1] - 1, seed[0] - 1]:
        # Nearest region pixel, first in column-major order on ties, as MATLAB's find.
        cx, cy = np.nonzero(region.T)
        if cx.size == 0:
            return label
        k = int(np.argmin(np.hypot(cx + 1.0 - point[0], cy + 1.0 - point[1])))
        seed = np.array([cx[k] + 1, cy[k] + 1])
    seeds = np.zeros_like(region)
    seeds[seed[1] - 1, seed[0] - 1] = True
    geodesic = _geodesic_distance(region, seeds)
    long_labels = valid[long]
    reach = np.array([geodesic[labels == k].min() for k in long_labels])
    k = int(np.argmin(reach))
    if np.isfinite(reach[k]):
        label = int(long_labels[k])
    return label


def _truss_partition(sc):
    """Assign every stem component to a truss; truss[k] is the truss of
    component k (0 for components below 100 px and for index 0).

    The partition does not depend on a click, so every click on the same truss
    selects the same components. First, components that lie in the same
    connected region of the relaxed stem mask form one unit. In a crate, the
    relaxed mask can join two complete trusses that touch, so a region with two
    or more components that span half of the region or more (maximum Feret
    diameter) is split into its components. Then, the units are grouped by the
    connected regions of touching fruits and stems. Within such a region, units
    that span at least 0.3 times the largest unit are trusses; smaller units,
    typically calyxes or pedicels detached from their peduncle, join the truss
    that they reach first along a path through the region. On the supplied
    images, detached calyxes spanned at most 0.26 and trusses in crates at
    least 0.41 times the largest unit.
    """
    areas, hj, wj = sc["areas"], sc["hj"], sc["wj"]
    n = areas.size - 1
    truss = np.zeros(n + 1, np.int64)
    valid = np.flatnonzero(areas >= 100)
    valid = valid[valid > 0]
    if valid.size == 0:
        return truss

    def feret(ids):
        return _max_feret_diameter(ids // hj + 1.0, ids % hj + 1.0)

    pixels = _pixel_lists(sc["labels_f"], n)
    extent = np.zeros(n + 1)
    extent[valid] = [feret(pixels[c]) for c in valid]
    relaxed_f = sc["relaxed_labels"].ravel(order="F")
    group = np.zeros(n + 1, np.int64)
    for c in valid:
        group[c] = int(np.argmax(np.bincount(relaxed_f[pixels[c]])))
    group_pixels = _pixel_lists(relaxed_f, int(relaxed_f.max()))
    unit = np.zeros(n + 1, np.int64)
    units = 0
    for g in np.unique(group[valid]):
        comps = valid[group[valid] == g]
        if g == 0 or np.count_nonzero(extent[comps] >= 0.5 * feret(group_pixels[g])) >= 2:
            unit[comps] = units + np.arange(1, comps.size + 1)
            units += comps.size
        else:
            units += 1
            unit[comps] = units
    unit_f = unit[sc["labels_f"]]
    unit_pixels = _pixel_lists(unit_f, units)
    unit_extent = np.zeros(units + 1)
    unit_extent[1:] = [feret(unit_pixels[u]) for u in range(1, units + 1)]
    blob_map, _ = _label_column_major(sc["selection_fruit"] | sc["stem"])
    blob_f = blob_map.ravel(order="F")
    unit_blob = np.zeros(units + 1, np.int64)
    for u in range(1, units + 1):
        unit_blob[u] = int(np.argmax(np.bincount(blob_f[unit_pixels[u]])))
    owner = np.arange(units + 1)
    unit_map = unit_f.reshape(wj, hj).T
    for b in np.unique(unit_blob[1:]):
        members = np.flatnonzero(unit_blob == b)
        members = members[members > 0]
        anchors = members[unit_extent[members] >= 0.3 * unit_extent[members].max()]
        others = np.setdiff1d(members, anchors)
        if others.size == 0:
            continue
        if anchors.size == 1:
            owner[others] = anchors[0]
            continue
        # Geodesic Voronoi partition of this region among its anchor units,
        # computed within the bounding box of the region.
        rows, cols = np.nonzero(blob_map == b)
        r0, c0 = rows.min(), cols.min()
        box = (slice(r0, rows.max() + 1), slice(c0, cols.max() + 1))
        in_blob = blob_map[box] == b
        unit_box = unit_map[box]
        nearest = np.full(in_blob.shape, np.inf, np.float32)
        closest = np.zeros(in_blob.shape, np.int64)
        for u in anchors:
            # bwdistgeodesic returns single precision; ties go to the first anchor.
            d = _geodesic_distance(in_blob, unit_box == u).astype(np.float32)
            closer = d < nearest
            nearest[closer] = d[closer]
            closest[closer] = u
        box_h, box_w = in_blob.shape
        for u in others:
            ids = unit_pixels[u]
            r, c = ids % hj - r0, ids // hj - c0
            keep = (r >= 0) & (r < box_h) & (c >= 0) & (c < box_w)
            if not keep.any():
                continue
            # The first minimum in column-major order, as MATLAB's min over find.
            reach = nearest[r[keep], c[keep]]
            i = int(np.argmin(reach))
            if np.isfinite(reach[i]):
                owner[u] = closest[r[keep][i], c[keep][i]]
    truss[valid] = owner[unit[valid]]
    return truss


def _pixel_lists(labels_f, n):
    """Column-major linear indices of the pixels of labels 0..n, each list in
    ascending order, from the column-major flattened label image labels_f."""
    order = np.argsort(labels_f, kind="stable")
    bounds = np.searchsorted(labels_f[order], np.arange(n + 2))
    return [order[bounds[k]:bounds[k + 1]] for k in range(n + 1)]


def _nearest_containing(labels, distance, inside):
    """The nearest of the labels whose hull contains the point, else the nearest."""
    distance = distance.copy()
    if inside.any():
        distance[~inside] = np.inf
    return int(labels[int(np.argmin(distance))])


def _max_feret_diameter(x, y):
    """Largest distance between two pixel corners of the region with pixel
    centers (x, y), as MATLAB's regionprops MaxFeretDiameter."""
    corners = np.concatenate([np.column_stack((x + dx, y + dy))
                              for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)])
    hull = cv2.convexHull(corners.astype(np.float32))[:, 0, :].astype(np.float64)
    return float(np.max(np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2)))


@numba.njit(cache=True)
def _geodesic_distance(region, seeds):
    """Quasi-Euclidean geodesic distance within region from the seed pixels
    that lie in region, as MATLAB's bwdistgeodesic: 8-connected paths with
    steps of 1 and sqrt(2). Pixels that cannot be reached are infinite."""
    h, w = region.shape
    dist = np.full((h, w), np.inf)
    heap = [(0.0, 0, 0)]
    heap.pop()
    for r in range(h):
        for c in range(w):
            if seeds[r, c] and region[r, c]:
                dist[r, c] = 0.0
                heap.append((0.0, r, c))
    diagonal = np.sqrt(2.0)
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not region[rr, cc]:
                    continue
                nd = d + (diagonal if dr != 0 and dc != 0 else 1.0)
                if nd < dist[rr, cc]:
                    dist[rr, cc] = nd
                    heapq.heappush(heap, (nd, rr, cc))
    return dist


def _grasp_at_point(sc, truss_point, options, output_file):
    """Place the grasp on the peduncle next to truss_point (1-based original
    pixels), instead of searching the truss for a free peduncle end."""
    xy_scale = sc["xy_scale"]
    target = (truss_point - 0.5) / xy_scale + 0.5
    point = np.full(2, np.nan)
    direction = np.full(2, np.nan)
    if sc["point_graph"] is not None:
        neighbors, degree, pixel_ids = sc["point_graph"]
        sc = dict(sc, neighbors=neighbors, degree=degree, pixel_ids=pixel_ids)
    node, path = _snap_to_stem(sc, target)
    if path is None or path.size < 3:
        info = _new_info(output_file, truss_point, 0)
        warnings.warn("No stem found near the selected point. Returning NaNs.",
                      RuntimeWarning, stacklevel=4)
    else:
        hj = sc["hj"]
        ids = sc["pixel_ids"]
        label = int(sc["labels_f"][ids[node]])
        info = _new_info(output_file, truss_point, label)
        xy = np.column_stack((ids[path] // hj + 1.0, ids[path] % hj + 1.0))
        diameter = 2 * np.median(sc["radius_f"][ids[path]])
        rows, cols = np.nonzero(sc["labels"] == label)
        center = np.array([cols.mean() + 1.0, rows.mean() + 1.0])
        point, direction, placement = _place_at_point(
            xy, diameter, target, center, sc["edge_mask"], sc["stem"], xy_scale)
        info.update(placement)
        info["status"] = "at_point"
        info["arrowEnd"] = point + options["arrow_length"] * direction
    if options["save_diagnostics"]:
        _add_diagnostics(info, sc, [info["trussComponent"]])
    return point, direction, info


def _snap_to_stem(sc, target):
    """Skeleton node on the peduncle near target ([x, y], working pixels), and
    the straight skeleton path through it.

    A click next to the peduncle may lie closer to a pedicel or a sepal, and
    pedicels can be nearly as thick as the peduncle. The peduncle, however,
    continues straight through its junctions, whereas a pedicel ends at its
    calyx or turns sharply into the peduncle. Candidates are the nearest nodes
    of the junction-free skeleton segments that come within two stem diameters
    of the nearest segment, excluding segments thinner than half the thickest
    (sepal tips). From each, the straightest path is followed for about four
    diameters to both sides, and the covered fraction of that length is
    computed. Near a cut end, the peduncle covers only one side, and a pedicel
    can be nearly as thick as the peduncle. The candidate with the largest
    product of path radius, covered fraction and squared path straightness
    (chord over arc length) therefore wins; among candidates within 5% of the
    best product, the nearest one. The straightness factor penalizes a path
    that runs from a pedicel through a junction into the peduncle. The path
    radius is the median over the junction-free nodes of the whole path, since
    the radius of a short segment is inflated by the adjacent junctions.
    """
    from scipy.sparse.csgraph import connected_components
    hj, wj = sc["hj"], sc["wj"]
    ids, degree, neighbors = sc["pixel_ids"], sc["degree"], sc["neighbors"]
    xy = np.column_stack((ids // hj + 1.0, ids % hj + 1.0))
    free = np.flatnonzero(degree <= 2)
    distance = np.hypot(xy[free, 0] - target[0], xy[free, 1] - target[1])
    near = np.flatnonzero(distance <= 0.04 * max(hj, wj))
    if near.size == 0:
        return None, None
    # Junction-free segments: components of the skeleton graph without junctions.
    position = np.full(ids.size, -1)
    position[free] = np.arange(free.size)
    src = np.repeat(free, [neighbors[i].size for i in free])
    dst = np.concatenate([neighbors[i] for i in free]) if free.size else np.zeros(0, np.int64)
    keep = position[dst] >= 0
    graph = sparse.csr_matrix((np.ones(keep.sum()), (position[src[keep]], position[dst[keep]])),
                              shape=(free.size, free.size))
    _, segments = connected_components(graph, directed=False)
    radius = sc["radius_f"][ids[free]].astype(np.float64)
    order = np.argsort(segments, kind="stable")
    bounds = np.flatnonzero(np.diff(segments[order])) + 1
    thickness = np.array([np.median(radius[g]) for g in np.split(order, bounds)])
    near = near[distance[near] <= distance[near].min() + 4 * thickness[segments[near]].max()]
    near = near[thickness[segments[near]] >= 0.5 * thickness[segments[near]].max()]
    # The nearest node of each remaining segment, in order of distance.
    near = near[np.argsort(distance[near], kind="stable")]
    _, first = np.unique(segments[near], return_index=True)
    first = np.sort(first)
    candidates = free[near[first]]
    candidate_radius = thickness[segments[near[first]]].copy()
    coverage = np.zeros(candidates.size)
    straightness = np.zeros(candidates.size)
    paths = []
    for k, node in enumerate(candidates):
        reach = 8 * candidate_radius[k]  # About 4 diameters.
        steps = max(6, int(np.floor(4 * candidate_radius[k] + 0.5)))  # About 2 diameters.
        path = _straight_path(neighbors, xy, node, reach, steps)
        paths.append(path)
        arc_length = np.sum(np.linalg.norm(np.diff(xy[path], axis=0), axis=1))
        coverage[k] = min(1.0, arc_length / (2 * reach))
        straightness[k] = np.linalg.norm(xy[path[-1]] - xy[path[0]]) / max(arc_length, _EPS)
        body = path[degree[path] <= 2]
        candidate_radius[k] = np.median(sc["radius_f"][ids[body]])
    support = candidate_radius * coverage * straightness ** 2
    k = int(np.flatnonzero(support >= 0.95 * support.max())[0])
    return int(candidates[k]), paths[k]


def _straight_path(neighbors, xy, node, reach, steps):
    """Skeleton nodes along the straightest continuation through node, up to
    reach pixels of arc length to both sides, ordered along the stem. steps is
    the number of nodes over which headings are measured at junctions."""
    nb = neighbors[node]
    if nb.size == 0:
        return np.array([node])
    # Leave node in the two most opposite directions, or in one at an endpoint.
    first, second, best = nb[0], None, np.inf
    for a in range(nb.size - 1):
        for b in range(a + 1, nb.size):
            u = xy[nb[a]] - xy[node]
            w = xy[nb[b]] - xy[node]
            c = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))
            if c < best:
                best, first, second = c, nb[a], nb[b]
    visited = np.zeros(xy.shape[0], bool)
    visited[node] = True
    side_a = _walk_straight(neighbors, xy, node, first, reach, steps, visited)
    path = side_a[::-1] + [node]
    if second is not None:
        path += _walk_straight(neighbors, xy, node, second, reach, steps, visited)
    return np.array(path, dtype=np.int64)


def _walk_straight(neighbors, xy, start, nxt, reach, steps, visited):
    """Follow the skeleton from start through nxt (visited is updated in place).

    At a junction, continue along the branch whose direction, steps nodes
    ahead, deviates least from the heading over the last steps nodes. Measured
    over about two stem diameters, these directions are not dominated by the
    local kinks of the skeleton at the junction. Stops after reach pixels of
    arc length, at an end, or where every continuation turns by more than 60
    degrees.
    """
    path = [nxt]
    visited[nxt] = True
    history = [start, nxt]
    travelled = np.linalg.norm(xy[nxt] - xy[start])
    while travelled < reach:
        current = path[-1]
        options = [j for j in neighbors[current] if not visited[j]]
        if not options:
            break
        step = options[0]
        if len(options) > 1:
            heading = xy[current] - xy[history[max(0, len(history) - steps)]]
            alignment = np.full(len(options), -np.inf)
            for i, j in enumerate(options):
                ahead = _look_ahead(neighbors, current, j, visited, steps)
                d = xy[ahead] - xy[current]
                alignment[i] = np.dot(d, heading) / (np.linalg.norm(d) * np.linalg.norm(heading))
            i = int(np.argmax(alignment))
            if alignment[i] < np.cos(np.pi / 3):
                break
            step = options[i]
        visited[step] = True
        travelled += np.linalg.norm(xy[step] - xy[current])
        path.append(step)
        history.append(step)
    return path


def _look_ahead(neighbors, start, nxt, visited, steps):
    """Node reached by following the unique continuation from nxt for up to steps."""
    node, previous = nxt, start
    for _ in range(steps):
        options = [j for j in neighbors[node] if j != previous and not visited[j]]
        if len(options) != 1:
            break
        previous, node = node, options[0]
    return node


def _place_at_point(xy, diameter, target, center, mask, strict_mask, xy_scale):
    """Center a curve along the skeleton path xy and place the grasp at its
    point nearest to target.

    xy, target and center are in working pixels; the results in original
    pixels. The direction points away from center, the center of the truss's
    stem system, since a point inside the peduncle has no cut end to face.
    """
    mask = mask.astype(np.float64)
    strict_mask = strict_mask.astype(np.float64)
    offsets = _section_offsets(diameter)
    curve = _centered_curve(xy, 0.0, diameter, offsets, mask, strict_mask, single=False)
    curve = (curve - 0.5) * xy_scale + 0.5
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))))
    arc, unique_rows = np.unique(arc, return_index=True)
    curve = curve[unique_rows]
    distance = _nearest_arc(arc, curve, (target - 0.5) * xy_scale + 0.5)
    p = _interp1(arc, curve, distance)
    v = _grasp_direction(arc, curve, distance, diameter * xy_scale.mean())
    if np.dot(v, p - ((center - 0.5) * xy_scale + 0.5)) < 0:
        v = -v
    # Final transverse correction uses the fitted direction at the grasp.
    pw = (p - 0.5) / xy_scale + 0.5
    tw = v / xy_scale
    tw = tw / np.linalg.norm(tw)
    pw, edges = _center_section(pw, tw, diameter, offsets, mask, strict_mask)
    p = (pw - 0.5) * xy_scale + 0.5
    return p, v, dict(centerline=curve, crossSectionEdges=(edges - 0.5) * xy_scale + 0.5)


def _nearest_arc(arc, curve, q):
    """Arc length of the point of the polyline curve nearest to q."""
    a = curve[:-1]
    d = curve[1:] - a
    t = np.clip(np.sum((q - a) * d, axis=1) / np.maximum(np.sum(d ** 2, axis=1), _EPS), 0, 1)
    i = int(np.argmin(np.linalg.norm(a + t[:, None] * d - q, axis=1)))
    return arc[i] + t[i] * (arc[i + 1] - arc[i])


def _draw_selection(I, p):
    """Mark a truss-selection point with a cross, in place."""
    line_width = max(2, int(np.floor(max(I.shape[0], I.shape[1]) / 650 + 0.5)))
    arm = 5 * line_width
    _paint_line(I, p - np.array([arm, arm]), p + np.array([arm, arm]), line_width, SELECTION_COLOR)
    _paint_line(I, p - np.array([arm, -arm]), p + np.array([arm, -arm]), line_width, SELECTION_COLOR)


def write_tomato_png(I, filename):
    """Write a lossless RGB8 PNG with a fixed Sub filter and zlib level 1.

    Port of writeTomatoPNG.m. Pixel values are unchanged; files can be
    larger than those of a standard PNG encoder.
    """
    if I.dtype != np.uint8 or I.ndim != 3 or I.shape[2] != 3:
        raise ValueError("Expected a uint8 RGB image.")
    h, w = I.shape[:2]
    rows = I.reshape(h, 3 * w)
    # PNG Sub uses the same channel one pixel to the left; uint8 wraps mod 256.
    scanlines = np.empty((h, 3 * w + 1), np.uint8)
    scanlines[:, 0] = 1
    scanlines[:, 1:4] = rows[:, :3]
    np.subtract(rows[:, 3:], rows[:, :-3], out=scanlines[:, 4:])
    compressed = _parallel_zlib(scanlines)

    header = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    with open(filename, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        for kind, data in ((b"IHDR", header), (b"IDAT", compressed), (b"IEND", b"")):
            f.write(len(data).to_bytes(4, "big"))
            f.write(kind)
            f.write(data)
            f.write((zlib.crc32(data, zlib.crc32(kind)) & 0xFFFFFFFF).to_bytes(4, "big"))


def _parallel_zlib(data, level=1):
    """Return a single zlib stream of data, compressed in parallel slices.

    Each slice is raw-deflated independently and ends on a full flush, so the
    concatenated slices form one valid deflate stream (as pigz does). zlib
    releases the GIL, so threads give a near-linear speedup. The compressed
    size grows by a few kilobytes because slices share no history.
    """
    buffer = memoryview(data).cast("B")
    n_slices = max(1, min(os.cpu_count() or 1, 16, len(buffer) // (1 << 20)))
    step = -(-len(buffer) // n_slices)

    def compress(i):
        compressor = zlib.compressobj(level, zlib.DEFLATED, -15)
        last = i == n_slices - 1
        return (compressor.compress(buffer[i * step:(i + 1) * step])
                + compressor.flush(zlib.Z_FINISH if last else zlib.Z_FULL_FLUSH))

    with ThreadPoolExecutor(n_slices) as pool:
        parts = list(pool.map(compress, range(n_slices)))
    # zlib header for deflate with a 32 KiB window and fastest compression.
    return b"".join([b"\x78\x01", *parts, zlib.adler32(buffer).to_bytes(4, "big")])


def _main(argv=None):
    import argparse
    import time
    parser = argparse.ArgumentParser(
        description="Detect the tomato peduncle grasp point and save <name>_grasp.png.")
    parser.add_argument("image", help="input RGB image")
    parser.add_argument("--output", default=None, help="output path (default: ./<name>_grasp.png)")
    parser.add_argument("--grasp-distance", type=float, default=None,
                        help="arc distance from the cut face, in original pixels")
    parser.add_argument("--arrow-length", type=float, default=150)
    parser.add_argument("--legacy", action="store_true",
                        help="reproduce the initial version (no adjacent-end junctions, local direction)")
    parser.add_argument("--truss-point", type=float, nargs=2, metavar=("X", "Y"), default=None,
                        help="1-based pixel on the truss to be grasped")
    parser.add_argument("--select", action="store_true",
                        help="click on the truss to be grasped in a window")
    parser.add_argument("--multiple", action="store_true",
                        help="with --select, click several trusses one by one (Enter finishes)")
    parser.add_argument("--grasp-at-point", action="store_true",
                        help="grasp the peduncle at the indicated point")
    args = parser.parse_args(argv)
    timer = time.perf_counter()
    point, direction, info = find_tomato_grasp(
        args.image, output_file=args.output, grasp_distance=args.grasp_distance,
        arrow_length=args.arrow_length, legacy=args.legacy, truss_point=args.truss_point,
        select_truss=args.select, multiple_trusses=args.multiple,
        grasp_at_point=args.grasp_at_point)
    elapsed = time.perf_counter() - timer
    infos = info if isinstance(info, list) else [info]
    points = np.atleast_2d(point)
    directions = np.atleast_2d(direction)
    for p, v, r in zip(points, directions, infos):
        print(f"status:    {r['status']}")
        print(f"point:     [{p[0]:.3f} {p[1]:.3f}]  (1-based x, y in pixels)")
        print(f"direction: [{v[0]:.6f} {v[1]:.6f}]")
    print(f"output:    {infos[0]['outputFile']}")
    print(f"time:      {elapsed:.3f} s")
    if args.select:
        import matplotlib.pyplot as plt
        plt.show()  # Keep the result window open until it is closed.


if __name__ == "__main__":
    _main()
