# Excluded model weights

These files are excluded from git via `.gitignore` (too large / near GitHub's
100MB per-file hard limit) and must be placed manually to run the pipeline:

- `grasp/src/detect_truss_obb/weights/best.pt` (87MB)
- `grasp/src/determine_grasp_candidates_oriented_keypoint/weights/best.pt` (154MB)
- `grasp/src/choose_grasp_pose_from_candidates/weights/depth_image_encoder.pth` (4.1MB)

TODO: document where these came from (training run / download location) and
how to regenerate them.
