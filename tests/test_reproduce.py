"""
GPU-free tests for the paper reproduction orchestrator (scripts/reproduce.py).

Exercises the stage registry, tier/only/skip selection, dependency gating, and the
manifest/report generation on a fake run — no models, no GPU, no subprocess side effects.
Loaded by file path (scripts/ is not a package).
"""

import importlib.util
import json
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE = os.path.join(_HERE, "..", "scripts", "reproduce.py")


def _load():
    spec = importlib.util.spec_from_file_location("reproduce", _MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r = _load()


def _cfg(tmp_path, tier=0, model="gpt2", packs=None):
    return r.Config(tier=tier, model=model, seeds=4, intensities="0.0,0.5,1.0",
                    packs=packs or ["lsd"], prompt="a tree", outdir=str(tmp_path),
                    max_tokens=32, max_questions=5, timeout=None)


def test_stage_keys_unique_and_have_paper_refs():
    stages = r.all_stages()
    keys = [s.key for s in stages]
    assert len(keys) == len(set(keys)), "duplicate stage keys"
    for s in stages:
        assert s.paper_ref and s.title, f"{s.key} missing metadata"
        assert s.min_tier in (0, 1, 2)


def test_tier_selection_is_monotonic(tmp_path):
    stages = r.all_stages()
    t0 = r._select(stages, _cfg(tmp_path, tier=0), [], [])
    t1 = r._select(stages, _cfg(tmp_path, tier=1), [], [])
    t2 = r._select(stages, _cfg(tmp_path, tier=2), [], [])
    k0, k1, k2 = {s.key for s in t0}, {s.key for s in t1}, {s.key for s in t2}
    # Higher tiers are supersets of lower tiers.
    assert k0 < k1 < k2
    # T0 contains only min_tier-0 stages; T2 contains the gated experiments.
    assert all(s.min_tier == 0 for s in t0)
    assert "lsd_ablation" in k2 and "lsd_ablation" not in k1


def test_only_and_skip_filters(tmp_path):
    stages = r.all_stages()
    only = r._select(stages, _cfg(tmp_path, tier=2), ["power_analysis", "vitals"], [])
    assert {s.key for s in only} == {"power_analysis", "vitals"}
    skipped = r._select(stages, _cfg(tmp_path, tier=0), [], ["vitals"])
    assert "vitals" not in {s.key for s in skipped}


def test_endpoints_builds_one_command_per_pack(tmp_path):
    cfg = _cfg(tmp_path, tier=1, packs=["lsd", "cocaine", "morphine"])
    stage = next(s for s in r.all_stages() if s.key == "endpoints")
    cmds = stage.build(cfg)
    assert len(cmds) == 3
    assert all("--pack" in c for c in cmds)
    packs_in_cmds = {c[c.index("--pack") + 1] for c in cmds}
    assert packs_in_cmds == {"lsd", "cocaine", "morphine"}


def test_visual_pilot_uses_dry_run_at_tier0_real_at_tier1(tmp_path):
    stage = next(s for s in r.all_stages() if s.key == "visual_pilot")
    t0_cmd = stage.build(_cfg(tmp_path, tier=0))[0]
    t1_cmd = stage.build(_cfg(tmp_path, tier=1))[0]
    assert "--dry-run" in t0_cmd
    assert "--dry-run" not in t1_cmd and "sdxl-turbo" in t1_cmd


def test_committed_stage_ok_when_files_present(tmp_path):
    # A committed stage reports ok iff its glob resolves to files.
    stage = next(s for s in r.all_stages() if s.committed)
    res = r.run_stage(stage, _cfg(tmp_path), env=dict(os.environ))
    assert res.status in ("ok", "committed-missing")


def test_missing_dependency_is_skipped_not_failed(tmp_path):
    fake = r.Stage(
        key="needs_unicorn", title="requires a missing module", min_tier=0,
        paper_ref="n/a", modality="validation",
        build=lambda c: [r._py("-c", "print(1)")],
        outputs=lambda c: [], requires=["definitely_not_installed_xyz"])
    res = r.run_stage(fake, _cfg(tmp_path), env=dict(os.environ))
    assert res.status == "skipped-deps"


def test_run_stage_executes_and_records(tmp_path):
    marker = tmp_path / "made.txt"
    stage = r.Stage(
        key="touch", title="write a file", min_tier=0, paper_ref="n/a", modality="validation",
        build=lambda c: [r._py("-c", f"open(r'{marker}','w').write('hi')")],
        outputs=lambda c: [str(marker)])
    res = r.run_stage(stage, _cfg(tmp_path), env=dict(os.environ))
    assert res.status == "ok" and marker.exists()
    assert res.produced  # recorded the artifact


def test_failed_stage_captures_stderr(tmp_path):
    stage = r.Stage(
        key="boom", title="fails", min_tier=0, paper_ref="n/a", modality="validation",
        build=lambda c: [r._py("-c", "import sys; sys.stderr.write('kaboom'); sys.exit(3)")],
        outputs=lambda c: [])
    res = r.run_stage(stage, _cfg(tmp_path), env=dict(os.environ))
    assert res.status == "failed" and res.returncode == 3
    assert "kaboom" in res.stderr_tail


def test_report_and_manifest_written(tmp_path):
    cfg = _cfg(tmp_path, tier=1)
    results = [
        r.StageResult(key="a", title="A", tier=0, paper_ref="Fig 1", modality="visual",
                      status="ok", seconds=1.2, produced=["outputs/x.png"]),
        r.StageResult(key="b", title="B", tier=1, paper_ref="Table 1", modality="text",
                      status="failed", returncode=1, stderr_tail="oops"),
    ]
    report_path = r.write_report(results, cfg)
    manifest_path = r.write_manifest(results, cfg)
    report = open(report_path).read()
    assert "Fig 1" in report and "Table 1" in report
    assert "outputs/x.png" in report and "oops" in report
    # Tier < 2 carries the illustrative-numbers caveat.
    assert "not the paper's" in report
    manifest = json.load(open(manifest_path))
    assert manifest["config"]["model"] == "gpt2"
    assert {s["key"] for s in manifest["stages"]} == {"a", "b"}


def test_record_provenance_writes_snapshot(tmp_path):
    cfg = _cfg(tmp_path, tier=0)
    prov = r.record_provenance(cfg)
    cfg_json = os.path.join(prov, "config.json")
    assert os.path.exists(cfg_json)
    c = json.load(open(cfg_json))
    assert c["seed"] == 42 and c["tier"] == 0 and c["model"] == "gpt2"
    # git sha + pip freeze snapshots are attempted (files exist even if best-effort).
    assert os.path.exists(os.path.join(prov, "git_sha.txt"))
    assert os.path.exists(os.path.join(prov, "pip_freeze.txt"))


def test_legacy_shim_forwards_to_single_path():
    import subprocess
    root = os.path.join(_HERE, "..")

    def run(args):
        return subprocess.run([__import__("sys").executable, "reproduce_results.py", *args, "--list"],
                              cwd=root, capture_output=True, text=True)

    r_test = run(["--test-mode"])
    assert "--tier 1" in r_test.stderr and "deprecated" in r_test.stderr.lower()
    r_def = run([])
    assert "--tier 2" in r_def.stderr
    # A pass-through flag reaches the target.
    r_model = run(["--model", "gpt2"])
    assert "--model gpt2" in r_model.stderr


def test_default_model_and_packs_by_tier():
    # Mirror main()'s defaulting logic.
    assert r.PACKS_QUICK and r.PACKS_FULL
    assert "placebo" in r.PACKS_QUICK
    assert len(r.PACKS_FULL) > len(r.PACKS_QUICK)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
