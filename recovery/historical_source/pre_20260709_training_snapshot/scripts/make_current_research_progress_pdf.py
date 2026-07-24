from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports" / "paper_protocol_summary"
DEFAULT_STEM = "current_research_progress"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY)
    parser.add_argument("--out-stem", default=DEFAULT_STEM)
    args = parser.parse_args()

    summary = args.summary_dir
    summary.mkdir(parents=True, exist_ok=True)
    report = build_report(summary)
    md_path = summary / f"{args.out_stem}.md"
    pdf_path = summary / f"{args.out_stem}.pdf"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    render_pdf(report, pdf_path)
    print(md_path)
    print(pdf_path)


def build_report(summary: Path) -> dict[str, Any]:
    closure = read_json(summary / "ten_hour_closure_report.json")
    audit = read_json(summary / "objective_completion_audit.json")
    active = read_json(summary / "active_training_watch_report.json")
    queue = read_json(summary / "queue_recovery_report.json")
    fair = read_json(summary / "fair_comparison_report.json")
    roadmap = read_json(summary / "topvenue_innovation_roadmap.json")
    module = read_json(summary / "module_retention_report.json")
    fair_dynamic = collect_fair_rows()
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "closure": closure,
        "audit": audit,
        "active": active,
        "queue": queue,
        "fair": fair,
        "roadmap": roadmap,
        "module": module,
        "fair_dynamic": fair_dynamic,
    }


def collect_fair_rows() -> list[dict[str, Any]]:
    root = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
    pairs = [
        ("RoadSaW", "single_roadsaw_full_faf", "baseline_single_roadsaw_global_convnext"),
        ("RSCD", "single_rscd_full_faf", "baseline_single_rscd_global_convnext"),
        ("RoadSC", "single_roadsc_full_faf", "baseline_single_roadsc_global_convnext"),
    ]
    rows = []
    for dataset, faf_name, base_name in pairs:
        faf = read_json(root / faf_name / "detailed_test.json")
        base = read_json(root / base_name / "detailed_test.json")
        faf_eval = read_json(root / faf_name / "evaluate_test.json")
        base_eval = read_json(root / base_name / "evaluate_test.json")
        faf_friction = task_metric(faf, "friction", "macro_f1")
        base_friction = task_metric(base, "friction", "macro_f1")
        faf_risk = task_metric(faf, "risk", "macro_f1")
        base_risk = task_metric(base, "risk", "macro_f1")
        rows.append(
            {
                "dataset": dataset,
                "faf_status": "complete" if faf else ("eval_only" if faf_eval else "missing"),
                "baseline_status": "complete" if base else ("eval_only" if base_eval else "missing"),
                "faf_friction_f1": faf_friction,
                "baseline_friction_f1": base_friction,
                "delta_friction_f1": diff(faf_friction, base_friction),
                "faf_risk_f1": faf_risk,
                "baseline_risk_f1": base_risk,
                "delta_risk_f1": diff(faf_risk, base_risk),
                "faf_low_recall": low_recall(faf),
                "baseline_low_recall": low_recall(base),
                "delta_low_recall": diff(low_recall(faf), low_recall(base)),
                "faf_raw_coverage": mu_metric(faf, "coverage"),
                "baseline_raw_coverage": mu_metric(base, "coverage"),
                "delta_raw_coverage": diff(mu_metric(faf, "coverage"), mu_metric(base, "coverage")),
                "faf_width": mu_metric(faf, "width_mean"),
                "baseline_width": mu_metric(base, "width_mean"),
            }
        )
    return rows


def task_metric(report: dict[str, Any], task: str, metric: str) -> Any:
    if not isinstance(report, dict):
        return None
    return ((report.get("tasks") or {}).get(task) or {}).get(metric)


def low_recall(report: dict[str, Any]) -> Any:
    if not isinstance(report, dict):
        return None
    low = report.get("low_friction_detection") or {}
    if low.get("applicable") is False:
        return None
    return low.get("recall")


def mu_metric(report: dict[str, Any], metric: str) -> Any:
    if not isinstance(report, dict):
        return None
    return (report.get("mu_interval") or {}).get(metric)


def diff(a: Any, b: Any) -> Any:
    try:
        if a is None or b is None:
            return None
        return float(a) - float(b)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def pct(value: Any) -> str:
    if value is None or value == "-":
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def num(value: Any, digits: int = 4) -> str:
    if value is None or value == "-":
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    closure = report["closure"]
    audit = report["audit"]
    active = report["active"]
    queue = report["queue"]
    fair_rows = report.get("fair_dynamic") or (closure.get("single_dataset_fairness", []) if isinstance(closure, dict) else [])
    p0_rows = closure.get("p0_ablation", []) if isinstance(closure, dict) else []
    lodo_rows = closure.get("lodo", []) if isinstance(closure, dict) else []
    modules = closure.get("module_decisions", []) if isinstance(closure, dict) else []

    lines: list[str] = []
    lines.append("# 视觉摩擦可供性研究进展与实验证据")
    lines.append("")
    lines.append(f"生成时间：`{report['generated_at']}`")
    lines.append("")
    lines.append("## 结论边界")
    lines.append("")
    lines.append(
        "当前研究对象是基于公开道路图像标签的弱监督视觉摩擦可供性区间估计，"
        "不能表述为同步实测轮胎-路面摩擦系数回归。"
    )
    lines.append("")
    lines.append("## 当前运行")
    active_run = active.get("active", {}) if isinstance(active, dict) else {}
    lines.append(
        f"- 运行状态：`{active.get('verdict', '-')}`；当前任务：`{active_run.get('name', '-')}`；"
        f"epoch `{active_run.get('epoch', '-')}/{active_run.get('epochs', '-')}`，"
        f"step `{active_run.get('step', '-')}/{active_run.get('steps', '-')}`。"
    )
    lines.append(
        f"- 队列：完成 `{queue.get('num_complete', '-')}`，运行/部分 `{queue.get('num_partial', '-')}`，"
        f"缺失 `{queue.get('num_missing', '-')}`。"
    )
    latest = active.get("latest_completed_epoch", {}) if isinstance(active, dict) else {}
    if latest:
        lines.append(
            f"- 最近完成 epoch `{latest.get('epoch')}`：val loss `{num(latest.get('val_loss'))}`，"
            f"risk acc `{pct(latest.get('val_acc_risk'))}`，friction acc `{pct(latest.get('val_acc_friction'))}`，"
            f"raw coverage `{pct(latest.get('val_mu_interval_coverage'))}`。"
        )
    lines.append("")
    lines.extend(markdown_table(
        ["模块/方法", "状态", "friction F1", "risk F1", "低摩擦召回", "校准覆盖率", "最差数据集 F1", "决策"],
        [
            [
                r.get("method", "-"),
                r.get("status", "-"),
                pct(r.get("friction_f1")),
                pct(r.get("risk_f1")),
                pct(r.get("low_friction_recall")),
                pct(r.get("calibrated_coverage")),
                pct(r.get("worst_dataset_f1")),
                r.get("decision", "-"),
            ]
            for r in p0_rows
        ],
        "P0 消融核心结果",
    ))
    lines.extend(markdown_table(
        ["留出数据集", "状态", "friction F1", "risk F1", "低摩擦召回", "校准覆盖率", "宽度", "解释"],
        [
            [
                r.get("held_out", "-"),
                r.get("status", "-"),
                pct(r.get("friction_f1")),
                pct(r.get("risk_f1")),
                pct(r.get("low_friction_recall")),
                pct(r.get("calibrated_coverage")),
                num(r.get("width")),
                r.get("interpretation", "-"),
            ]
            for r in lodo_rows
        ],
        "LODO 跨数据集压力测试",
    ))
    lines.extend(markdown_table(
        ["数据集", "FAF", "ConvNeXt", "FAF friction", "Baseline friction", "差值", "FAF risk", "Baseline risk", "差值", "低召回差", "raw覆盖差"],
        [
            [
                r.get("dataset", "-"),
                r.get("faf_status", "-"),
                r.get("baseline_status", "-"),
                pct(r.get("faf_friction_f1")),
                pct(r.get("baseline_friction_f1")),
                pct(r.get("delta_friction_f1")),
                pct(r.get("faf_risk_f1")),
                pct(r.get("baseline_risk_f1")),
                pct(r.get("delta_risk_f1")),
                pct(r.get("delta_low_recall")),
                pct(r.get("delta_raw_coverage")),
            ]
            for r in fair_rows
        ],
        "同数据集公平对比",
    ))
    lines.append("## 算法流程")
    lines.append("")
    lines.append(
        "1. 输入道路图像 x，经 ConvNeXt 主干提取全局语义特征 h。"
        "2. PhysicsTexture 分支显式编码湿滑、低纹理、反光、粗糙度等视觉物理线索。"
        "3. 多任务头同时预测摩擦状态、风险状态、湿度、积雪、材质、平整度，并输出弱摩擦区间 I_mu(x)=[mu_min, mu_max]。"
        "4. 区间头用覆盖损失、端点损失、宽度约束和单调风险约束，让预测既能覆盖弱标签区间，又不过度变宽。"
        "5. EvidenceField/ROI/segmentation-style 候选路线把证据从整图分类推进到局部道路区域证据，目标是减少数据集捷径。"
    )
    lines.append("")
    lines.append("## 模块取舍")
    lines.append("")
    for item in modules:
        lines.append(f"- `{item.get('module', '-')}`：`{item.get('decision', '-')}`。{item.get('evidence', '-')}")
    lines.append("")
    lines.append("## 目前能说")
    lines.append("")
    for item in audit.get("allowed_claims", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 目前不能说")
    lines.append("")
    for item in audit.get("not_allowed_yet", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 下一步")
    lines.append("")
    lines.append("1. 等 RSCD 和 RoadSC ConvNeXt 强基线完整结束，刷新公平对比表。")
    lines.append("2. 只对快速筛选胜出的候选算法做正式训练，避免把时间耗在已失败的全量融合路线。")
    lines.append("3. 若强基线追平或超过 FAF，论文主线转向可解释局部证据、区间安全性和跨数据集失败机理。")
    lines.append("4. 优先验证 segmentation-style region mixture、多查询局部证据和 masked consistency 是否能降低 dataset shortcut。")
    lines.append("")
    return "\n".join(lines)


def markdown_table(headers: list[str], rows: list[list[str]], title: str) -> list[str]:
    out = [f"## {title}", ""]
    if not rows:
        out += ["暂无完整行。", ""]
        return out
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    out.append("")
    return out


def render_pdf(report: dict[str, Any], pdf_path: Path) -> None:
    font = register_font()
    styles = make_styles(font)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    story: list[Any] = []
    add_p(story, "视觉摩擦可供性研究进展与实验证据", styles["title"])
    add_p(story, f"生成时间：{report['generated_at']}", styles["small"])
    add_h(story, "结论边界", styles)
    add_p(
        story,
        "当前研究对象是基于公开道路图像标签的弱监督视觉摩擦可供性区间估计，"
        "不能表述为同步实测轮胎-路面摩擦系数回归。",
        styles["body"],
    )

    active = report["active"]
    active_run = active.get("active", {}) if isinstance(active, dict) else {}
    latest = active.get("latest_completed_epoch", {}) if isinstance(active, dict) else {}
    queue = report["queue"]
    add_h(story, "当前运行", styles)
    add_p(
        story,
        f"运行状态：{active.get('verdict', '-')}；当前任务：{active_run.get('name', '-')}；"
        f"epoch {active_run.get('epoch', '-')}/{active_run.get('epochs', '-')}，"
        f"step {active_run.get('step', '-')}/{active_run.get('steps', '-')}。"
        f"队列完成 {queue.get('num_complete', '-')}，运行/部分 {queue.get('num_partial', '-')}，缺失 {queue.get('num_missing', '-')}。",
        styles["body"],
    )
    if latest:
        add_p(
            story,
            f"最近完成 epoch {latest.get('epoch')}：val loss {num(latest.get('val_loss'))}，"
            f"risk acc {pct(latest.get('val_acc_risk'))}，friction acc {pct(latest.get('val_acc_friction'))}，"
            f"raw coverage {pct(latest.get('val_mu_interval_coverage'))}。",
            styles["body"],
        )

    closure = report["closure"]
    add_table_section(
        story,
        "P0 消融核心结果",
        ["方法", "friction F1", "risk F1", "低摩擦召回", "校准覆盖", "最差 F1", "决策"],
        [
            [
                r.get("method", "-"),
                pct(r.get("friction_f1")),
                pct(r.get("risk_f1")),
                pct(r.get("low_friction_recall")),
                pct(r.get("calibrated_coverage")),
                pct(r.get("worst_dataset_f1")),
                r.get("decision", "-"),
            ]
            for r in closure.get("p0_ablation", [])
        ],
        styles,
    )
    add_table_section(
        story,
        "LODO 跨数据集压力测试",
        ["留出", "friction F1", "risk F1", "校准覆盖", "宽度", "解释"],
        [
            [
                r.get("held_out", "-"),
                pct(r.get("friction_f1")),
                pct(r.get("risk_f1")),
                pct(r.get("calibrated_coverage")),
                num(r.get("width")),
                r.get("interpretation", "-"),
            ]
            for r in closure.get("lodo", [])
        ],
        styles,
    )
    fair_pdf_rows = report.get("fair_dynamic") or closure.get("single_dataset_fairness", [])
    add_table_section(
        story,
        "同数据集公平对比",
        ["数据集", "FAF friction", "Base friction", "差值", "FAF risk", "Base risk", "差值", "低召回差", "raw覆盖差"],
        [
            [
                r.get("dataset", "-"),
                pct(r.get("faf_friction_f1")),
                pct(r.get("baseline_friction_f1")),
                pct(r.get("delta_friction_f1")),
                pct(r.get("faf_risk_f1")),
                pct(r.get("baseline_risk_f1")),
                pct(r.get("delta_risk_f1")),
                pct(r.get("delta_low_recall")),
                pct(r.get("delta_raw_coverage")),
            ]
            for r in fair_pdf_rows
        ],
        styles,
    )

    add_h(story, "算法流程", styles)
    for item in [
        "输入道路图像 x，经 ConvNeXt 主干提取全局语义特征 h。",
        "PhysicsTexture 分支显式编码湿滑、低纹理、反光、粗糙度等视觉物理线索。",
        "多任务头预测摩擦状态、风险状态、湿度、积雪、材质、平整度，并输出弱摩擦区间 I_mu(x)=[mu_min, mu_max]。",
        "区间头使用覆盖、端点、宽度和单调风险约束，让预测既覆盖弱标签区间，又不过度变宽。",
        "后续 EvidenceField/ROI/segmentation-style 路线把证据从整图分类推进到局部道路区域证据，目标是减少数据集捷径。",
    ]:
        add_p(story, item, styles["body"])

    add_h(story, "模块取舍", styles)
    for item in closure.get("module_decisions", []):
        add_p(story, f"{item.get('module', '-')}：{item.get('decision', '-')}。{item.get('evidence', '-')}", styles["body"])

    audit = report["audit"]
    add_h(story, "目前能说", styles)
    for item in audit.get("allowed_claims", []):
        add_p(story, f"- {item}", styles["body"])
    add_h(story, "目前不能说", styles)
    for item in audit.get("not_allowed_yet", []):
        add_p(story, f"- {item}", styles["body"])
    add_h(story, "下一步", styles)
    for item in [
        "等 RSCD 和 RoadSC ConvNeXt 强基线完整结束，刷新公平对比表。",
        "只对快速筛选胜出的候选算法做正式训练，避免把时间耗在已失败的全量融合路线。",
        "若强基线追平或超过 FAF，论文主线转向可解释局部证据、区间安全性和跨数据集失败机理。",
        "优先验证 segmentation-style region mixture、多查询局部证据和 masked consistency 是否能降低 dataset shortcut。",
    ]:
        add_p(story, f"- {item}", styles["body"])
    doc.build(story)


def add_table_section(story: list[Any], title: str, headers: list[str], rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> None:
    add_h(story, title, styles)
    if not rows:
        add_p(story, "暂无完整行。", styles["body"])
        return
    data = [[Paragraph(escape(str(cell)), styles["table"]) for cell in headers]]
    data += [[Paragraph(escape(str(cell)), styles["table"]) for cell in row] for row in rows]
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#12355b")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d3df")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.extend([table, Spacer(1, 8)])


def register_font() -> str:
    for font_path in [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont("CNFont", str(font_path)))
            return "CNFont"
    return "Helvetica"


def make_styles(font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleCN", parent=base["Title"], fontName=font, fontSize=17, leading=23, alignment=TA_CENTER, wordWrap="CJK"),
        "h": ParagraphStyle("HCN", parent=base["Heading1"], fontName=font, fontSize=12.5, leading=17, textColor=colors.HexColor("#12355b"), spaceBefore=9, spaceAfter=5, wordWrap="CJK"),
        "body": ParagraphStyle("BodyCN", parent=base["BodyText"], fontName=font, fontSize=9.2, leading=14, spaceAfter=5, wordWrap="CJK"),
        "small": ParagraphStyle("SmallCN", parent=base["BodyText"], fontName=font, fontSize=8.2, leading=12, textColor=colors.HexColor("#4b5563"), alignment=TA_CENTER, spaceAfter=8, wordWrap="CJK"),
        "table": ParagraphStyle("TableCN", parent=base["BodyText"], fontName=font, fontSize=7.0, leading=9, wordWrap="CJK"),
    }


def add_h(story: list[Any], text: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph(escape(text), styles["h"]))


def add_p(story: list[Any], text: str, style: ParagraphStyle) -> None:
    story.append(Paragraph(escape(str(text)), style))
    story.append(Spacer(1, 2))


if __name__ == "__main__":
    main()
