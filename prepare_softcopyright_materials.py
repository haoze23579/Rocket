from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SourceFileStat:
    rel_path: str
    line_count: int


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "软著材料"
CODE_OUT_DIR = OUT_DIR / "鉴别材料"
EVIDENCE_DIR = OUT_DIR / "证据材料"


def iter_source_files() -> Iterable[Path]:
    include_dirs = ["sim", "prediction", "planning", "evaluation", "visualization"]
    include_files = [
        "run_experiments.py",
        "record_demo.py",
        "demo_auto.py",
        "demo_auto_3d.py",
        "demo_manual.py",
        "demo_manual_3d.py",
    ]

    seen: set[Path] = set()
    for d in include_dirs:
        for p in sorted((ROOT / d).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            if p not in seen:
                seen.add(p)
                yield p

    for f in include_files:
        p = ROOT / f
        if p.exists() and p not in seen:
            seen.add(p)
            yield p


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def build_merged_code_lines(files: list[Path]) -> tuple[list[str], list[SourceFileStat]]:
    merged: list[str] = []
    stats: list[SourceFileStat] = []

    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        lines = read_lines(p)
        stats.append(SourceFileStat(rel_path=rel, line_count=len(lines)))

        merged.append(f"# ===== FILE BEGIN: {rel} =====")
        for i, line in enumerate(lines, start=1):
            merged.append(f"{i:04d}: {line}")
        merged.append(f"# ===== FILE END: {rel} =====")
        merged.append("")

    return merged, stats


def paginate(lines: list[str], lines_per_page: int = 50) -> list[list[str]]:
    pages: list[list[str]] = []
    for i in range(0, len(lines), lines_per_page):
        pages.append(lines[i : i + lines_per_page])
    return pages


def write_pages(path: Path, pages: list[list[str]], title: str) -> None:
    out: list[str] = [title, ""]
    for idx, page in enumerate(pages, start=1):
        out.append(f"================ 第 {idx:03d} 页 ================")
        out.extend(page)
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_evidence_files() -> list[Path]:
    candidates = [
        ROOT / "paper.md",
        ROOT / "paper.docx",
        ROOT / "paper_v4.docx",
        ROOT / "model_arch.png",
        ROOT / "run_experiments.py",
        ROOT / "run_recordings.ps1",
        ROOT / "experiment_log.txt",
        ROOT / "reports" / "exp20260303_catches.png",
        ROOT / "reports" / "exp20260303_efficiency.png",
        ROOT / "reports" / "exp20260303_delta_v.png",
        ROOT / "reports" / "success_rate_vs_speed.png",
        ROOT / "reports" / "prediction_comparison.png",
        ROOT / "reports" / "speed_heatmap.png",
        ROOT / "reports" / "videos" / "lstm.mp4",
        ROOT / "reports" / "videos" / "physics.mp4",
        ROOT / "reports" / "videos" / "reactive.mp4",
        ROOT.parent / "公共目录" / "experiment_results" / "orbit_experiment_results_20260303.csv",
        ROOT.parent / "公共目录" / "videos" / "baseline_orbit_simulator_20260303.mp4",
        ROOT.parent / "公共目录" / "videos" / "experiment_orbit_simulator_20260303.mp4",
    ]
    return [p for p in candidates if p.exists()]


def copy_evidence(files: list[Path]) -> list[tuple[Path, Path]]:
    copied: list[tuple[Path, Path]] = []
    for src in files:
        if src.is_relative_to(ROOT):
            rel = src.relative_to(ROOT)
            dst = EVIDENCE_DIR / "项目内证据" / rel
        else:
            rel = src.relative_to(ROOT.parent)
            dst = EVIDENCE_DIR / "项目外证据" / rel

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        copied.append((src, dst))
    return copied


def write_manifest(
    stats: list[SourceFileStat],
    merged_lines: list[str],
    total_pages: int,
    first_pages: int,
    last_pages: int,
    copied: list[tuple[Path, Path]],
) -> None:
    manifest = OUT_DIR / "06_材料清单与校验清单.md"
    total_lines = sum(s.line_count for s in stats)

    rows = [
        "# 软著材料清单与校验清单",
        "",
        "## 一、源程序规模",
        f"- 统计文件数：{len(stats)}",
        f"- 统计源代码总行数（.py）：{total_lines}",
        f"- 合并后行数（含文件分隔标记）：{len(merged_lines)}",
        f"- 总页数（每页50行）：{total_pages}",
        f"- 已导出前{first_pages}页与后{last_pages}页鉴别材料",
        "",
        "## 二、源程序文件明细",
        "",
        "| 序号 | 文件 | 行数 |",
        "|---:|---|---:|",
    ]
    for idx, s in enumerate(stats, start=1):
        rows.append(f"| {idx} | `{s.rel_path}` | {s.line_count} |")

    rows.extend(
        [
            "",
            "## 三、证据材料明细（已复制）",
            "",
            "| 序号 | 原始文件 | 复制到 | SHA256 |",
            "|---:|---|---|---|",
        ]
    )

    for idx, (src, dst) in enumerate(copied, start=1):
        rows.append(
            f"| {idx} | `{src}` | `{dst}` | `{sha256(dst)}` |"
        )

    rows.extend(
        [
            "",
            "## 四、鉴别材料文件",
            "",
            "- `鉴别材料/源程序_合并全文_每页50行.txt`",
            "- `鉴别材料/源程序_前30页_每页50行.txt`",
            "- `鉴别材料/源程序_后30页_每页50行.txt`",
            "",
            "## 五、建议最终打包目录",
            "",
            "- `01_申请信息表_待填.md`",
            "- `02_软件说明书_软著版.md`",
            "- `03_操作手册_软著版.md`",
            "- `04_官方规则摘要.md`",
            "- `05_提交打包清单_最终版.md`",
            "- `鉴别材料/`（3个txt）",
            "- `证据材料/`（CSV、图表、视频、论文等）",
        ]
    )
    manifest.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CODE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    files = list(iter_source_files())
    merged_lines, stats = build_merged_code_lines(files)
    pages = paginate(merged_lines, lines_per_page=50)
    total_pages = len(pages)

    first_n = min(30, total_pages)
    last_n = min(30, total_pages)

    all_pages_path = CODE_OUT_DIR / "源程序_合并全文_每页50行.txt"
    first_pages_path = CODE_OUT_DIR / "源程序_前30页_每页50行.txt"
    last_pages_path = CODE_OUT_DIR / "源程序_后30页_每页50行.txt"

    write_pages(all_pages_path, pages, "源程序鉴别材料（合并全文）")
    write_pages(first_pages_path, pages[:first_n], "源程序鉴别材料（前30页）")
    write_pages(last_pages_path, pages[-last_n:], "源程序鉴别材料（后30页）")

    evidence_files = collect_evidence_files()
    copied = copy_evidence(evidence_files)

    write_manifest(
        stats=stats,
        merged_lines=merged_lines,
        total_pages=total_pages,
        first_pages=first_n,
        last_pages=last_n,
        copied=copied,
    )

    print(f"Done. Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
