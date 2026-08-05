# Third-party checkouts

Clone these **into this directory** (or export the env vars). Nothing outside
this repository is assumed.

```bash
./scripts/bootstrap_third_party.sh
# or:
git clone --recursive https://github.com/GoldenGait/FARM-Project.git third_party/FARM-Project
git clone https://github.com/Kodifly/ss-3dgs.git third_party/ss-3dgs
```

| Path | Upstream | Env override |
|---|---|---|
| `third_party/FARM-Project` | [GoldenGait/FARM-Project](https://github.com/GoldenGait/FARM-Project) | `FARM_PROJECT_ROOT` |
| `third_party/ss-3dgs` | [Kodifly/ss-3dgs](https://github.com/Kodifly/ss-3dgs) (private) | `SS3DGS_ROOT` |

Weights stay in `FARM-Project/models` (or `SCENE_GRAPH_MODEL_DIR`). Run that
repo’s `bootstrap_models.sh` after clone. Install the vendored YOLOE fork from
`third_party/FARM-Project/third_party/yoloe`, not public ultralytics 8.4.x.
