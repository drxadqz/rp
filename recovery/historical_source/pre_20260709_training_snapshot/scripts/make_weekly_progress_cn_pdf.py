from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
SUMMARY = ROOT / "reports" / "paper_protocol_summary"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY)
    parser.add_argument("--out-stem", default="weekly_progress_clean")
    args = parser.parse_args()

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.summary_dir)
    md = render_markdown(report)
    md_path = args.summary_dir / f"{args.out_stem}.md"
    pdf_path = args.summary_dir / f"{args.out_stem}.pdf"
    md_path.write_text(md, encoding="utf-8")
    render_pdf(report, pdf_path)
    print(md_path)
    print(pdf_path)


def build_report(summary: Path) -> dict[str, Any]:
    queue = read_json(summary / "queue_recovery_report.json")
    closure = read_json(summary / "ten_hour_closure_report.json")
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "queue": queue,
        "closure": closure,
        "active": infer_active(queue),
        "p0": closure.get("p0_ablation", []) if isinstance(closure, dict) else [],
        "lodo": closure.get("lodo", []) if isinstance(closure, dict) else [],
        "fair": collect_fair_rows(),
        "datasets": collect_dataset_rows(),
    }


def infer_active(queue: dict[str, Any]) -> dict[str, Any]:
    rows = queue.get("queue_order") if isinstance(queue, dict) else []
    for row in rows or []:
        if row.get("status") == "running_or_partial":
            return row
    for row in rows or []:
        if row.get("status") == "missing":
            return {"name": row.get("name"), "status": "next_missing"}
    return {}


def collect_fair_rows() -> list[dict[str, Any]]:
    pairs = [
        ("RoadSaW", "single_roadsaw_full_faf", "baseline_single_roadsaw_global_convnext"),
        ("RSCD", "single_rscd_full_faf", "baseline_single_rscd_global_convnext"),
        ("RoadSC", "single_roadsc_full_faf", "baseline_single_roadsc_global_convnext"),
    ]
    rows = []
    for dataset, faf_name, base_name in pairs:
        faf = read_json(EXP_ROOT / faf_name / "detailed_test.json")
        base = read_json(EXP_ROOT / base_name / "detailed_test.json")
        rows.append(
            {
                "dataset": dataset,
                "faf_status": status_from(faf, EXP_ROOT / faf_name),
                "base_status": status_from(base, EXP_ROOT / base_name),
                "faf_friction": task_metric(faf, "friction", "macro_f1"),
                "base_friction": task_metric(base, "friction", "macro_f1"),
                "faf_risk": task_metric(faf, "risk", "macro_f1"),
                "base_risk": task_metric(base, "risk", "macro_f1"),
                "faf_low": low_recall(faf),
                "base_low": low_recall(base),
                "faf_cov": mu_metric(faf, "coverage"),
                "base_cov": mu_metric(base, "coverage"),
                "faf_width": mu_metric(faf, "width_mean"),
                "base_width": mu_metric(base, "width_mean"),
            }
        )
    return rows


def collect_dataset_rows() -> list[list[str]]:
    return [
        ["RSCD", "958,941 / 19,860 / 49,500", "路面状态、湿度、风险等弱标签", "样本最多，是单数据集强基线和主要鲁棒性检查对象"],
        ["RoadSaW", "15,360 / 4,812 / 1,728", "道路表面、湿滑/天气相关标签", "可验证湿滑、材质和风险可供性；高反光/近白样本是难点"],
        ["RoadSC", "4,374 / 1,467 / 573", "雪、冰、湿滑相关状态", "规模较小，适合做雪冰低摩擦场景补充验证"],
    ]


def status_from(detail: dict[str, Any], run_dir: Path) -> str:
    if detail:
        return "complete"
    if (run_dir / "evaluate_test.json").exists():
        return "eval_only"
    if (run_dir / "best.pt").exists():
        return "trained"
    return "missing"


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def num(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def delta(a: Any, b: Any) -> str:
    if a is None or b is None:
        return "-"
    return f"{(float(a) - float(b)) * 100:+.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    active = report["active"]
    lines = [
        "# 视觉道路摩擦可供性估计研究进展",
        "",
        f"生成时间：`{report['generated_at']}`",
        "",
        "## 1. 研究对象和边界",
        "",
        "本项目研究的是基于公开道路图像数据集的视觉摩擦可供性估计。由于现有公开数据集没有同步实测轮胎-路面摩擦系数，本研究不能声称直接回归真实摩擦系数，而是把 dry、wet、snow、ice、slush、material、roughness 等视觉标签映射为弱监督摩擦等级和摩擦区间。",
        "",
        "核心目标是：给定道路图像，预测摩擦等级、风险等级和一个弱摩擦区间，并尽量让模型的判断来自路面局部证据，而不是数据集风格、相机裁剪、色调或背景捷径。",
        "",
        "## 2. 当前运行状态",
        "",
        f"- 当前队列任务：`{active.get('name', '-')}`，状态：`{active.get('status', '-')}`。",
        f"- 队列完成数：`{report['queue'].get('num_complete', '-')}`；部分/运行中：`{report['queue'].get('num_partial', '-')}`；未开始：`{report['queue'].get('num_missing', '-')}`。",
        "",
        "## 3. 数据集和标签使用方式",
        "",
    ]
    lines += md_table(["数据集", "训练/验证/测试", "使用标签", "作用"], report["datasets"])
    lines += [
        "",
        "标签映射原则：干燥沥青/混凝土通常映射到高摩擦区间；湿路、积水、雪、冰、泥泞映射到更低摩擦区间；材质、粗糙度、湿度和雪冰标签作为辅助任务，帮助模型学习“为什么低摩擦”。这些标签是公开数据集可获得的信息，所以实验可以复现，不依赖自己采集车辆动力学数据。",
        "",
        "## 4. 算法流程",
        "",
        "输入图像记为 x。主干网络提取全局特征 h = f_theta(x)。PhysicsTexture 分支提取和湿滑、反光、粗糙、纹理稀疏相关的显式视觉物理线索 p = g_phi(x)。多任务头基于 z = concat(h, p) 同时预测摩擦等级、风险等级、湿度、积雪、材质、平整度，并输出摩擦区间 I_mu(x) = [mu_min(x), mu_max(x)]。",
        "",
        "最终训练目标可以写成：",
        "",
        "L = L_cls + lambda_aux L_aux + lambda_int L_interval + lambda_mono L_mono + lambda_cal L_cal。",
        "",
        "其中 L_cls 是摩擦等级和风险等级的交叉熵损失；L_aux 是湿度、雪冰、材质、平整度等辅助标签损失；L_interval 约束预测区间覆盖弱标签给出的摩擦范围；L_mono 约束高风险样本的摩擦上界不能异常偏高；L_cal 用于校准区间覆盖率，避免模型只给很窄但不可靠的区间。",
        "",
        "区间覆盖可以理解为：真实弱标签区间 [mu_l, mu_u] 应尽量落在模型预测区间 [mu_min, mu_max] 内，同时区间不能无限变宽。一个常用写法是：",
        "",
        "L_interval = max(0, mu_l - mu_min) + max(0, mu_max - mu_u) + beta (mu_max - mu_min)。",
        "",
        "这里 mu_l 和 mu_u 来自标签映射；beta 控制区间宽度惩罚。beta 太小会导致区间过宽，beta 太大会导致覆盖率下降。",
        "",
        "## 5. 当前核心实验结果",
        "",
    ]
    lines += md_table(
        ["数据集", "FAF状态", "基线状态", "FAF摩擦F1", "基线摩擦F1", "差值", "FAF风险F1", "基线风险F1", "差值", "低摩擦召回差", "原始覆盖差"],
        [
            [
                r["dataset"],
                r["faf_status"],
                r["base_status"],
                pct(r["faf_friction"]),
                pct(r["base_friction"]),
                delta(r["faf_friction"], r["base_friction"]),
                pct(r["faf_risk"]),
                pct(r["base_risk"]),
                delta(r["faf_risk"], r["base_risk"]),
                delta(r["faf_low"], r["base_low"]),
                delta(r["faf_cov"], r["base_cov"]),
            ]
            for r in report["fair"]
        ],
    )
    lines += [
        "",
        "严格判断：RoadSaW 上当前 FAF 比 ConvNeXt 强基线略好，说明显式物理纹理和多任务风险建模在湿滑/材质场景有价值；RSCD 上强 ConvNeXt 基线反而超过当前 FAF，尤其原始区间覆盖率高很多。这说明不能把当前完整 FAF 包装成全面优于强基线的最终算法，必须把论文主线转为“可解释局部证据、弱监督区间安全性、跨数据集失败机理与改进”。",
        "",
        "## 6. 已经验证出的模块结论",
        "",
    ]
    p0_rows = []
    for r in report["p0"]:
        p0_rows.append([
            r.get("method", "-"),
            pct(r.get("friction_f1")),
            pct(r.get("risk_f1")),
            pct(r.get("low_friction_recall")),
            pct(r.get("calibrated_coverage")),
            pct(r.get("worst_dataset_f1")),
            r.get("decision", "-"),
        ])
    lines += md_table(["模块", "摩擦F1", "风险F1", "低摩擦召回", "校准覆盖", "最差数据集F1", "判断"], p0_rows)
    lines += [
        "",
        "结论是：PhysicsTexture 是当前最值得保留的有效增量；FrictionSet、DG losses 和完整复杂融合没有形成稳定收益，需要迅速舍弃或重做；下一版应更轻、更局部、更像语义分割/区域证据建模，而不是堆叠更多全图模块。",
        "",
        "## 7. 创新点定位",
        "",
        "第一，研究对象从单纯分类改为摩擦可供性区间估计，输出既包含等级又包含不确定区间。第二，把 CV 中语义分割和区域证据思想迁移到道路摩擦：模型不只看整图，而要找到真正支持低摩擦判断的路面区域。第三，引入物理纹理线索，把反光、湿滑、粗糙度、低纹理等视觉物理因素显式建模。第四，用跨数据集失败分析约束论文叙事，不把多数据集混训当作当然有效，而是把域偏移本身作为要解决的问题。",
        "",
        "## 8. 主要问题",
        "",
        "当前最大问题有三个：一是 RSCD 上强基线已经超过当前 FAF，说明复杂模块存在负迁移；二是 LODO 跨数据集结果很差，说明模型强烈依赖数据集风格和标签体系；三是公开数据集缺少真实摩擦系数，只能做弱监督区间，论文表述必须严谨。",
        "",
        "## 9. 下一步",
        "",
        "短期先完成 RoadSC 强基线，刷新公平表；随后只保留 PhysicsTexture 和轻量局部证据模块，快速筛选 v23-v25 这类 segmentation-style / masked consistency 候选。若候选不能超过 RSCD 强基线，就舍弃完整 FAF 方案，改走“强基线 + 可解释局部证据 + 校准区间 + 域偏移诊断”的稳妥路线。中期需要补充可视化证据热力图、区间校准曲线、按湿度/材质/雪冰分组的错误分析，以及跨数据集标签映射合理性说明。",
    ]
    return "\n".join(lines) + "\n"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["无可用结果。"]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return out


def render_pdf(report: dict[str, Any], out: Path) -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    base = ParagraphStyle("cn", parent=styles["BodyText"], fontName="STSong-Light", fontSize=10.2, leading=15)
    title = ParagraphStyle("title", parent=base, fontSize=18, leading=24, alignment=TA_CENTER, spaceAfter=12)
    h1 = ParagraphStyle("h1", parent=base, fontSize=13.5, leading=18, textColor=colors.HexColor("#1f4e79"), spaceBefore=8, spaceAfter=6)
    small = ParagraphStyle("small", parent=base, fontSize=8.5, leading=12)

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.45 * cm, rightMargin=1.45 * cm, topMargin=1.35 * cm, bottomMargin=1.25 * cm)
    story: list[Any] = [Paragraph("视觉道路摩擦可供性估计研究进展", title), Paragraph(f"生成时间：{report['generated_at']}", small), Spacer(1, 8)]

    def add_heading(text: str) -> None:
        story.append(Paragraph(text, h1))

    def add_para(text: str) -> None:
        story.append(Paragraph(text, base))
        story.append(Spacer(1, 5))

    def add_table(rows: list[list[str]], widths: list[float]) -> None:
        wrapped = [[Paragraph(str(cell), small) for cell in row] for row in rows]
        table = Table(wrapped, colWidths=widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f2fb")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365d")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c7d9")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

    add_heading("1. 研究对象和边界")
    add_para("本项目研究的是基于公开道路图像数据集的视觉摩擦可供性估计。现有公开数据集没有同步实测轮胎-路面摩擦系数，因此本文应严谨表述为弱监督摩擦等级和摩擦区间估计。")
    add_para("核心目标是让模型根据路面局部视觉证据预测摩擦等级、风险等级和弱摩擦区间，而不是依赖数据集风格、相机裁剪、色调或背景捷径。")

    add_heading("2. 当前运行状态")
    active = report["active"]
    add_para(f"当前队列任务：{active.get('name', '-')}；状态：{active.get('status', '-')}。队列完成 {report['queue'].get('num_complete', '-')} 个，部分/运行中 {report['queue'].get('num_partial', '-')} 个，未开始 {report['queue'].get('num_missing', '-')} 个。")

    add_heading("3. 数据集和标签使用方式")
    add_table([["数据集", "训练/验证/测试", "使用标签", "作用"]] + report["datasets"], [2.3 * cm, 3.2 * cm, 4.0 * cm, 7.7 * cm])
    add_para("标签映射原则：干燥路面通常对应较高摩擦区间；湿路、积水、雪、冰、泥泞对应较低摩擦区间；材质、粗糙度、湿度和雪冰标签作为辅助任务，用来解释低摩擦来源。")

    add_heading("4. 算法流程和公式")
    add_para("输入图像记为 x。主干网络提取全局特征 h = f_theta(x)。PhysicsTexture 分支提取视觉物理线索 p = g_phi(x)。融合特征 z = concat(h, p) 同时预测摩擦等级、风险等级、辅助状态和摩擦区间 I_mu(x) = [mu_min(x), mu_max(x)]。")
    add_para("总损失：L = L_cls + lambda_aux L_aux + lambda_int L_interval + lambda_mono L_mono + lambda_cal L_cal。")
    add_para("变量解释：L_cls 是摩擦和风险分类损失；L_aux 是湿度、雪冰、材质、平整度等辅助损失；L_interval 是区间覆盖损失；L_mono 是风险单调性约束；L_cal 是校准项；lambda_* 是各损失权重。")
    add_para("区间损失：L_interval = max(0, mu_l - mu_min) + max(0, mu_max - mu_u) + beta (mu_max - mu_min)。其中 [mu_l, mu_u] 是由公开标签映射得到的弱摩擦区间，beta 控制区间宽度惩罚。")

    add_heading("5. 当前核心实验结果")
    fair_rows = [["数据集", "FAF状态", "基线状态", "FAF摩擦F1", "基线摩擦F1", "差值", "FAF风险F1", "基线风险F1", "差值", "覆盖差"]]
    for r in report["fair"]:
        fair_rows.append([
            r["dataset"], r["faf_status"], r["base_status"], pct(r["faf_friction"]), pct(r["base_friction"]), delta(r["faf_friction"], r["base_friction"]),
            pct(r["faf_risk"]), pct(r["base_risk"]), delta(r["faf_risk"], r["base_risk"]), delta(r["faf_cov"], r["base_cov"]),
        ])
    add_table(fair_rows, [1.5 * cm, 1.45 * cm, 1.45 * cm, 1.7 * cm, 1.7 * cm, 1.3 * cm, 1.7 * cm, 1.7 * cm, 1.3 * cm, 1.5 * cm])
    add_para("严格判断：RoadSaW 上 FAF 略优于强 ConvNeXt 基线；RSCD 上强 ConvNeXt 基线超过当前 FAF，尤其原始区间覆盖率明显更好。因此当前不能声称完整 FAF 全面领先，下一步必须做模块裁剪和局部证据化。")

    add_heading("6. 模块结论")
    p0_rows = [["模块", "摩擦F1", "风险F1", "低摩擦召回", "校准覆盖", "最差数据集F1", "判断"]]
    for r in report["p0"]:
        p0_rows.append([r.get("method", "-"), pct(r.get("friction_f1")), pct(r.get("risk_f1")), pct(r.get("low_friction_recall")), pct(r.get("calibrated_coverage")), pct(r.get("worst_dataset_f1")), r.get("decision", "-")])
    add_table(p0_rows, [3.5 * cm, 1.75 * cm, 1.75 * cm, 2.0 * cm, 2.0 * cm, 2.1 * cm, 4.0 * cm])
    add_para("PhysicsTexture 是当前最值得保留的有效模块；FrictionSet、DG losses 和完整复杂融合没有形成稳定收益，应快速舍弃或重做。")

    story.append(PageBreak())
    add_heading("7. 创新点定位")
    add_para("第一，把问题从普通道路状态分类推进到摩擦可供性区间估计。第二，把语义分割和区域证据思想迁移到道路摩擦，使模型学习低摩擦证据来自哪里。第三，引入视觉物理纹理线索，显式建模反光、湿滑、粗糙度和低纹理。第四，把跨数据集失败作为研究对象，反向约束算法设计和论文叙事。")

    add_heading("8. 主要问题")
    add_para("主要问题有三点：RSCD 上强基线超过当前 FAF，说明复杂模块存在负迁移；LODO 跨数据集结果差，说明数据集风格和标签体系偏移很强；公开数据集缺少真实摩擦系数，论文必须强调弱监督和可供性区间。")

    add_heading("9. 下一步")
    add_para("短期先完成 RoadSC 强基线并刷新公平表。随后只保留 PhysicsTexture 和轻量局部证据模块，快速筛选 segmentation-style region mixture、multi-query evidence 和 masked consistency 候选。若候选不能超过 RSCD 强基线，就转为强基线加可解释局部证据、区间校准和域偏移诊断的稳妥路线。")

    doc.build(story)


if __name__ == "__main__":
    main()
